import torch
import traceback

def is_close_my(a, a_ref, rtol=1e-3, atol=1e-3):
    try:
        torch.testing.assert_close(a, a_ref, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(e)
        # traceback.print_exc()
        traceback.print_stack()
        print("\n")
        return False