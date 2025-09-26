import ctypes
import os

print("---- Checking for CUDA Runtime Library ----")

# This is the core library for CUDA 12.x
cuda_dll_name = "cudart64_12.dll"

try:
    ctypes.WinDLL(cuda_dll_name)
    print(f"✅ SUCCESS: The main CUDA runtime library '{cuda_dll_name}' was found and loaded.")
    print("This suggests the PATH is likely correct and the issue is more subtle.")

except OSError as e:
    print(f"❌ FAILURE: Could not load the CUDA runtime library '{cuda_dll_name}'.")
    print("\nThis confirms the issue is that the CUDA directories are not in your system's PATH.")
    print("Please follow the solution below.")