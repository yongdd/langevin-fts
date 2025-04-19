import numpy as np

# LR compressor
class LR:
    def __init__(self, nx, lx, jk):
        self.nx = nx
        self.lx = lx
        self.jk = jk.copy()

        # self.jk_array_shape = nx.copy()
        # self.jk_array_shape[-1] = self.jk_array_shape[-1]//2 + 1
        # self.jk = np.zeros(self.jk_array_shape)

    def reset_count(self,):
        pass
        
    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        nx = self.nx
        w_diff = np.zeros_like(w_current)
        negative_h_deriv_k = np.fft.rfftn(np.reshape(negative_h_deriv, nx))/np.prod(nx)
        w_diff_k = negative_h_deriv_k/self.jk
        w_diff[0] = np.reshape(np.fft.irfftn(w_diff_k, nx), np.prod(nx))*np.prod(nx)

        return w_current + w_diff

# LRAM compressor
class LRAM:
    def __init__(self, lr, am):
        self.lr = lr
        self.am = am

    def reset_count(self,):
        self.am.reset_count()
        
    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        nx = self.lr.nx
        
        w_lr_old = w_current.copy()
        w_lr_new = np.reshape(self.lr.calculate_new_fields(w_lr_old, negative_h_deriv, old_error_level, error_level), [1, np.prod(nx)])
        w_diff = w_lr_new - w_lr_old
        w_new = np.reshape(self.am.calculate_new_fields(w_lr_new, w_diff, old_error_level, error_level), [1, np.prod(nx)])

        return w_new