#!/bin/bash
# Test suite verification script
# Copyright (c) 2025 Celaya Solutions AI Research Lab

echo "=========================================="
echo "NeuralChild Test Suite Verification"
echo "=========================================="
echo ""

cd /home/user/baby-llm/NeuralChild

echo "1. Checking test file syntax..."
python3 << 'PYTHON'
import sys
files = [
    'neuralchild/tests/__init__.py',
    'neuralchild/tests/conftest.py',
    'neuralchild/tests/test_config.py',
    'neuralchild/tests/test_schemas.py',
    'neuralchild/tests/test_networks.py',
    'neuralchild/tests/test_mind.py',
    'neuralchild/tests/test_mother.py',
    'neuralchild/tests/test_message_bus.py'
]

all_valid = True
for f in files:
    try:
        compile(open(f).read(), f, 'exec')
        print(f'  ✓ {f}')
    except SyntaxError as e:
        print(f'  ✗ {f}: {e}')
        all_valid = False

sys.exit(0 if all_valid else 1)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All test files have valid syntax!"
else
    echo ""
    echo "❌ Some test files have syntax errors"
    exit 1
fi

echo ""
echo "2. Counting test files and lines..."
wc -l neuralchild/tests/*.py | tail -1

echo ""
echo "3. Checking for copyright headers..."
for f in neuralchild/tests/*.py; do
    if grep -q "Celaya Solutions" "$f"; then
        echo "  ✓ $f has copyright header"
    else
        echo "  ✗ $f missing copyright header"
    fi
done

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. pip install -r requirements.txt"
echo "  2. pip install pytest pytest-cov"
echo "  3. pytest neuralchild/tests/ -v"
echo ""
