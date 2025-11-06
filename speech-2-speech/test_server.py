#!/usr/bin/env python3
"""
Simple test script to validate server setup
"""
import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        'fastapi',
        'uvicorn',
        'websockets',
        'numpy',
        'torch',
        'sounddevice',
        'faster_whisper',
        'transformers',
        'edge_tts',
        'shutil',
        'subprocess',
        'asyncio',
        'threading',
        'queue',
        'time',
        'traceback',
        'logging',
        'json',
        'base64'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"[OK] {module}")
        except ImportError as e:
            print(f"[FAIL] {module}: {e}")
            missing_modules.append(module)

    return missing_modules

def test_server_import():
    """Test if server.py can be imported"""
    try:
        import server
        print("[OK] server.py imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] server.py import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing server setup...")
    print("\n1. Testing module imports:")
    missing = test_imports()

    print("\n2. Testing server import:")
    server_ok = test_server_import()

    print("\n" + "="*50)
    if not missing and server_ok:
        print("[SUCCESS] All tests passed! Server is ready to run.")
        print("Run: python main.py")
    else:
        print("[ERROR] Some issues found:")
        if missing:
            print(f"Missing modules: {', '.join(missing)}")
            print("Install with: pip install -r requirements.txt")
        if not server_ok:
            print("Server module has import issues")

    sys.exit(0 if not missing and server_ok else 1)
