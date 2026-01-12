#!/usr/bin/env python
"""
Project cleanup script - removes debug files, caches, and unwanted artifacts.
Run this to clean up the project before distribution or final testing.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_project():
    """Remove debug files, caches, and temporary artifacts."""
    
    print("\n" + "=" * 70)
    print("PROJECT CLEANUP")
    print("=" * 70 + "\n")
    
    # Define cleanup patterns
    cleanup_items = [
        # Python caches
        ("__pycache__", "directory"),
        ("*.pyc", "files"),
        ("*.pyo", "files"),
        (".pytest_cache", "directory"),
        (".coverage", "file"),
        ("*.egg-info", "directory"),
        (".eggs", "directory"),
        
        # Jupyter/notebook caches
        (".ipynb_checkpoints", "directory"),
        ("*.ipynb_checkpoints", "directory"),
        
        # Virtual environment (optional - commented out for safety)
        # (".venv", "directory"),
        # ("venv", "directory"),
        
        # IDE caches
        (".vscode/__pycache__", "directory"),
        (".idea", "directory"),
        
        # Temporary/debug files
        ("debug_*.csv", "files"),
        ("temp_*", "files"),
        ("*.tmp", "files"),
    ]
    
    removed_count = 0
    
    # Cleanup Python caches recursively
    print("1️⃣ Removing Python caches...")
    for pyc in glob.glob("**/*.pyc", recursive=True):
        try:
            os.remove(pyc)
            print(f"   ✓ Removed: {pyc}")
            removed_count += 1
        except:
            pass
    
    for pyo in glob.glob("**/*.pyo", recursive=True):
        try:
            os.remove(pyo)
            print(f"   ✓ Removed: {pyo}")
            removed_count += 1
        except:
            pass
    
    # Cleanup __pycache__ directories
    for pycache in glob.glob("**/__pycache__", recursive=True):
        try:
            shutil.rmtree(pycache)
            print(f"   ✓ Removed: {pycache}/")
            removed_count += 1
        except:
            pass
    
    # Cleanup .pytest_cache
    if os.path.exists(".pytest_cache"):
        try:
            shutil.rmtree(".pytest_cache")
            print(f"   ✓ Removed: .pytest_cache/")
            removed_count += 1
        except:
            pass
    
    # Cleanup Jupyter checkpoints
    print("\n2️⃣ Removing Jupyter caches...")
    for checkpoint_dir in glob.glob("**/.ipynb_checkpoints", recursive=True):
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"   ✓ Removed: {checkpoint_dir}/")
            removed_count += 1
        except:
            pass
    
    # Cleanup temporary files
    print("\n3️⃣ Removing temporary files...")
    for temp_file in glob.glob("**/temp_*", recursive=True):
        try:
            if os.path.isfile(temp_file):
                os.remove(temp_file)
                print(f"   ✓ Removed: {temp_file}")
                removed_count += 1
        except:
            pass
    
    for tmp_file in glob.glob("**/*.tmp", recursive=True):
        try:
            os.remove(tmp_file)
            print(f"   ✓ Removed: {tmp_file}")
            removed_count += 1
        except:
            pass
    
    # Cleanup debug files
    print("\n4️⃣ Removing debug files...")
    for debug_file in glob.glob("**/debug_*", recursive=True):
        try:
            if os.path.isfile(debug_file):
                os.remove(debug_file)
                print(f"   ✓ Removed: {debug_file}")
                removed_count += 1
        except:
            pass
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✓ Cleanup Complete!")
    print(f"  Items removed: {removed_count}")
    print("=" * 70 + "\n")
    
    print("📋 Remaining structure:")
    print("  - Source code: src/")
    print("  - Data: data/handsigns/, data/dataset/, data/test_dataset/")
    print("  - Models: models/")
    print("  - Notebooks: notebooks/")
    print("  - Configuration: requirements.txt, README.md")
    print()

if __name__ == "__main__":
    try:
        cleanup_project()
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
