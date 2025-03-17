# compiled_model.py
import torch
import math
from .difflogic import LogicLayer, GroupSum
import tempfile
import subprocess
import shutil
import ctypes
import numpy as np
import numpy.typing
import time
from typing import Union

# Mapping dictionaries for generating C code.
BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0",
                        16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1",
                       16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {8: ctypes.c_int8, 16: ctypes.c_int16,
                   32: ctypes.c_int32, 64: ctypes.c_int64}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}

class CompiledLogicNet(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Sequential,
            device='cpu',
            num_bits=64,
            cpu_compiler='gcc',
            verbose=False,
    ):
        super(CompiledLogicNet, self).__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.cpu_compiler = cpu_compiler
        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [8, 16, 32, 64]

        if self.model is not None:
            layers = []
            self.num_inputs = None
            # The last layer must be GroupSum.
            assert isinstance(self.model[-1], GroupSum), 'Last layer must be GroupSum.'
            self.num_classes = self.model[-1].k
            first = True
            for layer in self.model:
                if isinstance(layer, LogicLayer):
                    if first:
                        self.num_inputs = layer.in_dim
                        first = False
                    # For each logic layer, use connectivity indices and hard negation flags.
                    neg_flags = (torch.sigmoid(layer.negation_logits) > 0.5).to(torch.uint8)
                    layers.append((layer.indices[0], layer.indices[1], neg_flags))
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose:
                        print('Skipping Flatten layer.')
                elif isinstance(layer, GroupSum):
                    if verbose:
                        print('Skipping GroupSum layer.')
                else:
                    raise ValueError(f'Unknown layer type: {type(layer)}')
            self.layers = layers
            if verbose:
                print('Parsed {} layers for compilation.'.format(len(layers)))
        self.lib_fn = None

    def get_inverter_gate_code(self, var1, var2, inv_a, inv_b):
        # If inv_a is true then invert var1; similarly for var2.
        a = f"~{var1}" if inv_a else var1
        b = f"~{var2}" if inv_b else var2
        res = f"({a} & {b})"
        if self.num_bits == 8:
            res = f"(char) {res}"
        elif self.num_bits == 16:
            res = f"(short) {res}"
        return res

    def get_layer_code(self, layer_a, layer_b, neg_flags, layer_id, prefix_sums):
        code = []
        neg_flags_np = neg_flags.cpu().numpy()  # shape: [out_dim, 2]
        for var_id, (gate_a, gate_b) in enumerate(zip(layer_a, layer_b)):
            inv_a = bool(neg_flags_np[var_id, 0])
            inv_b = bool(neg_flags_np[var_id, 1])
            if self.device == 'cpu' and layer_id == len(prefix_sums) - 1:
                a = f"v{prefix_sums[layer_id-1] + gate_a}"
                b = f"v{prefix_sums[layer_id-1] + gate_b}"
                gate_code = self.get_inverter_gate_code(a, b, inv_a, inv_b)
                code.append(f"\tout[{var_id}] = {gate_code};")
            else:
                if layer_id == 0:
                    a = f"inp[{gate_a}]"
                    b = f"inp[{gate_b}]"
                else:
                    a = f"v{prefix_sums[layer_id-1] + gate_a}"
                    b = f"v{prefix_sums[layer_id-1] + gate_b}"
                gate_code = self.get_inverter_gate_code(a, b, inv_a, inv_b)
                code.append(f"\tconst {BITS_TO_DTYPE[self.num_bits]} v{prefix_sums[layer_id] + var_id} = {gate_code};")
        return code

    def get_c_code(self):
        prefix_sums = [0]
        cur_count = 0
        for layer_a, layer_b, neg_flags in self.layers[:-1]:
            cur_count += len(layer_a)
            prefix_sums.append(cur_count)
        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "",
            f"void logic_gate_net({BITS_TO_DTYPE[self.num_bits]} const *inp, {BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ]
        for layer_id, (layer_a, layer_b, neg_flags) in enumerate(self.layers):
            code.extend(self.get_layer_code(layer_a, layer_b, neg_flags, layer_id, prefix_sums))
        code.append("}")
        num_neurons_ll = self.layers[-1][0].shape[0]
        log2_of_num_neurons_per_class_ll = math.ceil(math.log2(num_neurons_ll / self.num_classes + 1))
        code.append(fr"""
void apply_logic_gate_net (bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc({self.num_inputs} * sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc({num_neurons_ll} * sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp_o = malloc({log2_of_num_neurons_per_class_ll} * sizeof({BITS_TO_DTYPE[self.num_bits]}));
    
    for(size_t i = 0; i < len; ++i) {{
        for(size_t d = 0; d < {self.num_inputs}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(inp[i * {self.num_inputs} * {self.num_bits} + ({self.num_bits} - b - 1) * {self.num_inputs} + d]);
            }}
            inp_temp[d] = res;
        }}
        logic_gate_net(inp_temp, out_temp);
        for(size_t c = 0; c < {self.num_classes}; ++c) {{
            for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                out_temp_o[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            }}
            for(size_t a = 0; a < {self.layers[-1][0].shape[0] // self.num_classes}; ++a) {{
                {BITS_TO_DTYPE[self.num_bits]} carry = out_temp[c * {self.layers[-1][0].shape[0] // self.num_classes} + a];
                {BITS_TO_DTYPE[self.num_bits]} out_temp_o_d;
                for(int d = {log2_of_num_neurons_per_class_ll} - 1; d >= 0; --d) {{
                    out_temp_o_d  = out_temp_o[d];
                    out_temp_o[d] = carry ^ out_temp_o_d;
                    carry         = carry & out_temp_o_d;
                }}
            }}
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                const {BITS_TO_DTYPE[self.num_bits]} bit_mask = {BITS_TO_ONE_LITERAL[self.num_bits]} << b;
                {BITS_TO_DTYPE[32]} res = 0;
                for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                    res <<= 1;
                    res += !!(out_temp_o[d] & bit_mask);
                }}
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = res;
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}
""")
        return "\n".join(code)

    def compile(self, opt_level=1, save_lib_path=None, verbose=False):
        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c") as c_file:
                code = self.get_c_code()
                if verbose and len(code.split('\n')) <= 200:
                    print(code)
                c_file.write(code)
                c_file.flush()
                t_s = time.time()
                compiler_out = subprocess.run(
                    [
                        self.cpu_compiler,
                        "-shared",
                        "-fPIC",
                        "-O{}".format(opt_level),
                        "-o",
                        lib_file.name,
                        c_file.name,
                    ]
                )
                if compiler_out.returncode != 0:
                    raise RuntimeError(f'Compilation failed with code {compiler_out.returncode}')
                print('Compiling finished in {:.3f} seconds.'.format(time.time() - t_s))
            if save_lib_path is not None:
                shutil.copy(lib_file.name, save_lib_path)
                if verbose:
                    print('Library copied to', save_lib_path)
            lib = ctypes.cdll.LoadLibrary(lib_file.name)
            lib_fn = lib.apply_logic_gate_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]
        self.lib_fn = lib_fn

    @staticmethod
    def load(save_lib_path, num_classes, num_bits):
        self = CompiledLogicNet(None, num_bits=num_bits)
        self.num_classes = num_classes
        lib = ctypes.cdll.LoadLibrary(save_lib_path)
        lib_fn = lib.apply_logic_gate_net
        lib_fn.restype = None
        lib_fn.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        self.lib_fn = lib_fn
        return self

    def forward(
            self,
            x: Union[torch.BoolTensor, np.ndarray],
            verbose: bool = False
    ) -> torch.IntTensor:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
        pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
        x = np.concatenate([x, np.zeros((pad_len, x.shape[1]), dtype=bool)])
        if verbose:
            print('x.shape', x.shape)
        out = np.zeros(x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32])
        x = x.reshape(-1)
        self.lib_fn(x, out, batch_size_div_bits)
        out = torch.tensor(out).view(batch_size_div_bits * self.num_bits, self.num_classes)
        if pad_len > 0:
            out = out[:-pad_len]
        if verbose:
            print('out.shape', out.shape)
        return out

########################################################################################################################
class GroupSum(torch.nn.Module):
    """
    GroupSum module for aggregating logic gate outputs.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        from .packbitstensor import PackBitsTensor
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)
        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return f'k={self.k}, tau={self.tau}'
