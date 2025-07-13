import sys
import google.generativeai

print("--- Python Environment Diagnostics ---")
print(f"Python Executable: {sys.executable}")
print(f"Library Version:   {google.generativeai.__version__}")
print(f"Library Location:  {google.generativeai.__file__}")
print("------------------------------------")