
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import src.explainability...")
    import src.explainability
    print("Successfully imported src.explainability")
except Exception as e:
    print(f"Failed to import src.explainability: {e}")
    sys.exit(1)
