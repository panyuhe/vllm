#!/usr/bin/env python3
"""
Test runner for pooling tests - automatically discovers and runs all tests
while skipping bfloat16 tests that aren't supported on older GPUs.
"""

import os
import sys
import subprocess
import glob
import re
from pathlib import Path
from typing import List, Set, Tuple

def find_test_files() -> List[str]:
    """Find all test files in the pooling directory and related test files."""
    # Auto-discover pooling tests
    pooling_tests = glob.glob("vllm/tests/models/language/pooling/test_*.py")
    
    # Add specific additional test files
    additional_tests = [
        "vllm/tests/entrypoints/openai/test_embedding_dimensions.py",
        "vllm/tests/entrypoints/openai/test_score.py"
    ]
    
    # Filter to only existing files
    all_tests = pooling_tests + [f for f in additional_tests if os.path.exists(f)]
    
    return sorted(all_tests)

def discover_bfloat16_tests(test_files: List[str]) -> Set[str]:
    """Discover all bfloat16 tests that should be skipped."""
    bfloat16_tests = set()
    
    print("🔍 Discovering bfloat16 tests to skip...")
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
            
        try:
            # Run pytest collection to find all test names
            result = subprocess.run(
                ["pytest", "--collect-only", "-q", test_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Find test names containing bfloat16
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if '::' in line and 'bfloat16' in line:
                        # Extract just the test name
                        test_name = line.split()[0] if ' ' in line else line
                        bfloat16_tests.add(test_name)
                        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            print(f"⚠️  Warning: Could not collect tests from {test_file}: {e}")
            continue
    
    return bfloat16_tests

def reset_gpu_memory():
    """Reset GPU memory cache."""
    try:
        subprocess.run([
            "python", "-c", "import torch; torch.cuda.empty_cache()"
        ], check=True, capture_output=True)
        print("✅ GPU memory reset")
    except subprocess.SubprocessError:
        print("⚠️  Warning: Could not reset GPU memory")

def extract_test_results(log_content: str) -> List[str]:
    """Extract test results from log content (same logic as our summarizer)."""
    results = []
    lines = log_content.split('\n')
    
    # Find test cases with immediate status
    test_pattern = r'^(.+?::test_.*?(?:\[.*?\])?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)'
    
    for line in lines:
        line = line.strip()
        match = re.match(test_pattern, line)
        if match:
            test_name = match.group(1)
            status = match.group(2)
            if status != 'SKIPPED':  # Exclude skipped tests
                results.append(f"{test_name} - {status}")
    
    # Find test cases with delayed status
    test_with_logs_pattern = r'^(.+?::test_.*?(?:\[.*?\])?)(?:\s+INFO|\s+WARNING|\s+ERROR|$)'
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(test_with_logs_pattern, line)
        
        if match:
            test_name = match.group(1)
            
            # Skip if we already found this test
            if any(test_name in result for result in results):
                i += 1
                continue
            
            # Look ahead for status
            for j in range(i + 1, min(len(lines), i + 200)):
                status_line = lines[j].strip()
                
                if status_line in ['PASSED', 'FAILED', 'SKIPPED', 'ERROR', 'XFAIL', 'XPASS']:
                    if status_line != 'SKIPPED':
                        results.append(f"{test_name} - {status_line}")
                    break
                    
                if re.match(test_with_logs_pattern, status_line) and '::test_' in status_line:
                    break
        
        i += 1
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result in results:
        if result not in seen:
            seen.add(result)
            unique_results.append(result)
    
    return unique_results

def run_test_file(test_file: str, bfloat16_tests: Set[str]) -> Tuple[str, List[str]]:
    """Run a single test file and return the log file and results."""
    test_name = Path(test_file).stem
    log_file = f"test_output_{test_name}.log"
    
    print("=" * 60)
    print(f"📝 Running: {test_file}")
    print(f"📄 Output:  {log_file}")
    print(f"⏭️  Skipping: {len(bfloat16_tests)} bfloat16 tests")
    print("=" * 60)
    
    # Reset GPU memory
    reset_gpu_memory()
    
    # Build pytest command
    cmd = ["pytest", "-s", "-vvv"]
    
    # Add deselect arguments for bfloat16 tests
    for test in bfloat16_tests:
        cmd.extend(["--deselect", test])
    
    cmd.append(test_file)
    
    # Run pytest and capture output
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and file
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                output_lines.append(line)
            
            process.wait()
            log_content = ''.join(output_lines)
        
        # Extract results
        results = extract_test_results(log_content)
        return log_file, results
        
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return log_file, []

def write_summary_report(all_results: List[Tuple[str, str, List[str]]], summary_file: str):
    """Write the summary report to file."""
    with open(summary_file, 'w') as f:
        for test_file, log_file, results in all_results:
            f.write(f"\nSummary for {test_file}:\n")
            if results:
                for result in results:
                    f.write(f"{result}\n")
            else:
                f.write("No test results found\n")
            f.write("\n")

def write_skipped_tests(bfloat16_tests: Set[str], skipped_file: str):
    """Write the list of skipped bfloat16 tests to file."""
    with open(skipped_file, 'w') as f:
        for test in sorted(bfloat16_tests):
            f.write(f"{test}\n")

def main():
    print("🧪 Pooling Test Runner")
    print("=" * 60)
    
    # Find all test files
    test_files = find_test_files()
    print(f"📁 Found {len(test_files)} test files:")
    for tf in test_files:
        print(f"   {tf}")
    print()
    
    # Discover bfloat16 tests to skip
    bfloat16_tests = discover_bfloat16_tests(test_files)
    print(f"⏭️  Found {len(bfloat16_tests)} bfloat16 tests to skip")
    
    if bfloat16_tests:
        print("   Sample skipped tests:")
        for test in sorted(list(bfloat16_tests)[:5]):
            print(f"     {test}")
        if len(bfloat16_tests) > 5:
            print(f"     ... and {len(bfloat16_tests) - 5} more")
    print()
    
    # Initialize summary files
    summary_file = "summary_report.txt"
    skipped_file = "skipped_bfloat16_tests.txt"
    
    # Write skipped tests file
    write_skipped_tests(bfloat16_tests, skipped_file)
    
    # Run all tests
    all_results = []
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            continue
        
        log_file, results = run_test_file(test_file, bfloat16_tests)
        all_results.append((test_file, log_file, results))
    
    # Write summary report
    write_summary_report(all_results, summary_file)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("✅ All tests finished!")
    print("=" * 60)
    
    # Display summary
    print("\n📊 SUMMARY:")
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for test_file, log_file, results in all_results:
        passed = len([r for r in results if " - PASSED" in r])
        failed = len([r for r in results if " - FAILED" in r])
        total_tests += len(results)
        total_passed += passed
        total_failed += failed
        
        status_icon = "✅" if failed == 0 else "❌"
        print(f"{status_icon} {Path(test_file).name}: {passed} passed, {failed} failed")
    
    print(f"\n🎯 OVERALL: {total_passed} passed, {total_failed} failed, {total_tests} total")
    
    print(f"\n📝 Detailed logs: test_output_*.log")
    print(f"📋 Summary report: {summary_file}")
    print(f"⏭️  Skipped tests: {skipped_file}")
    
    print(f"\n💡 For clean summary, run:")
    print(f"   python test_log_summarizer.py test_output_*.log")
    
    # Exit with error code if any tests failed
    if total_failed > 0:
        print(f"\n❌ {total_failed} tests failed!")
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed!")

if __name__ == "__main__":
    main()
