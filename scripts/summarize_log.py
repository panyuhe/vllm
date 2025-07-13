#!/usr/bin/env python3
"""
Simple Test Result Extractor - Shows test case names and their status with validation and OOM detection
"""

import re
import argparse
import glob
import subprocess
import sys
from collections import Counter

def parse_short_test_summary(content):
    """Parse the 'short test summary info' section for detailed failure info."""
    summary_info = {}
    lines = content.split('\n')
    in_summary_section = False
    
    for line in lines:
        line = line.strip()
        
        # Check if we're entering the summary section
        if "short test summary info" in line:
            in_summary_section = True
            continue
        
        # Exit summary section when we hit the final summary line
        if in_summary_section and re.match(r'^=+\s*\d+.*=+$', line):
            break
            
        # Parse failed test lines in the summary section
        if in_summary_section and line.startswith('FAILED '):
            # Extract test name and error info
            match = re.match(r'FAILED\s+(.+?)\s+-\s+(.+)', line)
            if match:
                test_name = match.group(1)
                error_msg = match.group(2)
                summary_info[test_name] = error_msg
    
    return summary_info

def extract_oom_details(error_msg):
    """Extract memory details from OOM error message."""
    details = {}
    
    # Extract attempted allocation
    alloc_match = re.search(r'allocate ([\d.]+)\s*(MiB|GiB)', error_msg)
    if alloc_match:
        amount = float(alloc_match.group(1))
        unit = alloc_match.group(2)
        details['attempted'] = f"{amount} {unit}"
    
    # Extract free memory
    free_match = re.search(r'of which ([\d.]+)\s*(MiB|GiB) is free', error_msg)
    if free_match:
        amount = float(free_match.group(1))
        unit = free_match.group(2)
        details['free'] = f"{amount} {unit}"
    
    # Extract total capacity
    total_match = re.search(r'total capacity of ([\d.]+)\s*(GiB|MiB)', error_msg)
    if total_match:
        amount = float(total_match.group(1))
        unit = total_match.group(2)
        details['total'] = f"{amount} {unit}"
    
    # Extract memory in use
    used_match = re.search(r'this process has ([\d.]+)\s*(GiB|MiB) memory in use', error_msg)
    if used_match:
        amount = float(used_match.group(1))
        unit = used_match.group(2)
        details['used'] = f"{amount} {unit}"
    
    return details

def detect_cuda_oom_in_logs(log_content: str, test_name: str) -> bool:
    """Check if a test failure is due to CUDA OOM."""
    oom_patterns = [
        r'CUDA out of memory',
        r'OutOfMemoryError',
        r'RuntimeError.*CUDA out of memory',
        r'torch\.cuda\.OutOfMemoryError',
        r'CUDA_ERROR_OUT_OF_MEMORY',
        r'out of memory.*cuda',
        r'insufficient memory.*gpu'
    ]
    
    # First, try to find the test in the "short test summary info" section
    lines = log_content.split('\n')
    in_summary_section = False
    
    for line in lines:
        line = line.strip()
        
        # Check if we're entering the summary section
        if "short test summary info" in line:
            in_summary_section = True
            continue
        
        # Exit summary section when we hit the final summary line
        if in_summary_section and re.match(r'^=+\s*\d+.*=+$', line):
            break
            
        # Look for our test in the summary section
        if in_summary_section and test_name in line:
            # Check the rest of this line for OOM patterns
            for pattern in oom_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return True
    
    # Fallback: Look for OOM patterns in the log content around the test
    for i, line in enumerate(lines):
        if test_name in line and '::test_' in line:
            # Check next 100 lines for OOM patterns
            for j in range(i, min(len(lines), i + 100)):
                line_to_check = lines[j]
                for pattern in oom_patterns:
                    if re.search(pattern, line_to_check, re.IGNORECASE):
                        return True
            break
    
    return False

def parse_summary_line(content):
    """Parse the pytest summary line to extract counts"""
    summary_counts = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'error': 0,
        'warnings': 0
    }
    
    # Look for summary lines like "========== 7 passed, 26 skipped, 75808 warnings in 501.81s =========="
    summary_pattern = r'=+\s*(.+?)\s*=+'
    
    for line in content.split('\n'):
        line = line.strip()
        if re.match(summary_pattern, line):
            summary_text = re.match(summary_pattern, line).group(1)
            
            # Only process lines that contain test results
            if any(word in summary_text.lower() for word in ['passed', 'failed', 'skipped', 'error']):
                # Parse different result types
                passed_match = re.search(r'(\d+)\s+passed', summary_text)
                failed_match = re.search(r'(\d+)\s+failed', summary_text)
                skipped_match = re.search(r'(\d+)\s+skipped', summary_text)
                error_match = re.search(r'(\d+)\s+error', summary_text)
                warnings_match = re.search(r'(\d+)\s+warnings?', summary_text)
                
                if passed_match:
                    summary_counts['passed'] = int(passed_match.group(1))
                if failed_match:
                    summary_counts['failed'] = int(failed_match.group(1))
                if skipped_match:
                    summary_counts['skipped'] = int(skipped_match.group(1))
                if error_match:
                    summary_counts['error'] = int(error_match.group(1))
                if warnings_match:
                    summary_counts['warnings'] = int(warnings_match.group(1))
                
                return summary_counts, summary_text
    
    return summary_counts, None

def extract_test_results(content, include_skipped=False):
    """Extract test case results from log content with enhanced OOM detection"""
    results = []
    lines = content.split('\n')
    
    # Parse the short test summary for detailed failure info
    summary_info = parse_short_test_summary(content)
    
    # First, find all test case lines with immediate status (SKIPPED, etc.)
    test_pattern_same_line = r'^(.+?::test_.*?(?:\[.*?\])?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)'
    
    for line in lines:
        line = line.strip()
        match = re.match(test_pattern_same_line, line)
        if match:
            test_name = match.group(1)
            status = match.group(2)
            
            # Skip SKIPPED tests unless include_skipped is True
            if not include_skipped and status == 'SKIPPED':
                continue
                
            # Check for OOM if it's a failure
            if status in ['FAILED', 'ERROR']:
                # First check the summary info for this test
                if test_name in summary_info:
                    error_msg = summary_info[test_name]
                    if any(pattern in error_msg.lower() for pattern in ['cuda out of memory', 'outofmemoryerror', 'torch.outofmemoryerror']):
                        status = f"{status}_OOM"
                elif detect_cuda_oom_in_logs(content, test_name):
                    status = f"{status}_OOM"
                
            results.append(f"{test_name} - {status}")
    
    # Now find test cases that have their status after log output
    test_with_logs_pattern = r'^(.+?::test_.*?(?:\[.*?\])?)(?:\s+INFO|\s+WARNING|\s+ERROR|$)'
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(test_with_logs_pattern, line)
        
        if match:
            test_name = match.group(1)
            
            # Skip if we already found this test with immediate status
            if any(test_name in result for result in results):
                i += 1
                continue
            
            # Look ahead for the status (PASSED, FAILED, etc.)
            j = i + 1
            found_status = False
            
            # Look for status in the next several lines
            while j < min(len(lines), i + 200):
                status_line = lines[j].strip()
                
                # Check if this line contains just a status
                if status_line in ['PASSED', 'FAILED', 'SKIPPED', 'ERROR', 'XFAIL', 'XPASS']:
                    # Skip SKIPPED tests unless include_skipped is True
                    if not include_skipped and status_line == 'SKIPPED':
                        found_status = True
                        break
                        
                    # Check for OOM if it's a failure
                    status = status_line
                    if status in ['FAILED', 'ERROR']:
                        # First check the summary info for this test
                        if test_name in summary_info:
                            error_msg = summary_info[test_name]
                            if any(pattern in error_msg.lower() for pattern in ['cuda out of memory', 'outofmemoryerror', 'torch.outofmemoryerror']):
                                status = f"{status}_OOM"
                        elif detect_cuda_oom_in_logs(content, test_name):
                            status = f"{status}_OOM"
                        
                    results.append(f"{test_name} - {status}")
                    found_status = True
                    break
                
                # If we hit another test case line, stop looking
                if re.match(test_with_logs_pattern, status_line) and '::test_' in status_line:
                    break
                    
                j += 1
        
        i += 1
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result in results:
        if result not in seen:
            seen.add(result)
            unique_results.append(result)
    
    return unique_results

def count_parsed_results(results):
    """Count the parsed results by status"""
    counts = Counter()
    for result in results:
        if ' - ' in result:
            status = result.split(' - ')[-1]
            # Handle OOM as a separate category but also count as failed
            if '_OOM' in status:
                base_status = status.replace('_OOM', '').lower()
                counts['oom'] = counts.get('oom', 0) + 1
                counts[base_status] = counts.get(base_status, 0) + 1
            else:
                counts[status.lower()] = counts.get(status.lower(), 0) + 1
    return counts

def get_pytest_collection(test_path=None):
    """Run pytest --collect-only to get all available tests"""
    try:
        cmd = ['pytest', '--collect-only', '-q']
        if test_path:
            cmd.append(test_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse the collection output
            lines = result.stdout.split('\n')
            test_files = []
            
            for line in lines:
                line = line.strip()
                if '::test_' in line and not line.startswith('<'):
                    # Clean up the line to extract just the test name
                    test_name = line.split()[0] if ' ' in line else line
                    test_files.append(test_name)
            
            return test_files
        else:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

def process_file(filepath, validate=False, include_skipped=False):
    """Process a single log file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = extract_test_results(content, include_skipped)
        summary_counts, summary_text = parse_summary_line(content)
        parsed_counts = count_parsed_results(results)
        
        file_result = {
            'filepath': filepath,
            'results': results,
            'summary_counts': summary_counts,
            'summary_text': summary_text,
            'parsed_counts': parsed_counts,
            'validation': {}
        }
        
        # Validate parsed results against summary
        if summary_text:
            validation = {}
            total_summary = summary_counts['passed'] + summary_counts['failed'] + summary_counts['skipped'] + summary_counts['error']
            total_parsed = sum([v for k, v in parsed_counts.items() if k != 'oom'])  # Don't double-count OOM
            
            validation['summary_total'] = total_summary
            validation['parsed_total'] = total_parsed
            
            # Adjust validation based on whether we're including skipped tests
            if include_skipped:
                validation['counts_match'] = (
                    parsed_counts.get('passed', 0) == summary_counts['passed'] and
                    parsed_counts.get('failed', 0) == summary_counts['failed'] and
                    parsed_counts.get('skipped', 0) == summary_counts['skipped'] and
                    parsed_counts.get('error', 0) == summary_counts['error']
                )
            else:
                # When excluding skipped tests, only validate non-skipped counts
                validation['counts_match'] = (
                    parsed_counts.get('passed', 0) == summary_counts['passed'] and
                    parsed_counts.get('failed', 0) == summary_counts['failed'] and
                    parsed_counts.get('error', 0) == summary_counts['error']
                )
            
            file_result['validation'] = validation
        
        # Optional: validate against pytest collection
        if validate:
            # Try to figure out the test path from the log content
            test_path = None
            for result in results:
                if ' - ' in result:
                    test_name = result.split(' - ')[0]
                    if '::' in test_name:
                        test_path = test_name.split('::')[0]
                        break
            
            if test_path:
                all_tests = get_pytest_collection(test_path)
                if all_tests:
                    file_result['validation']['all_available_tests'] = len(all_tests)
                    file_result['validation']['pytest_collection'] = all_tests[:10]  # Show first 10 as sample
        
        return file_result
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def format_result_with_emoji(result):
    """Add emoji indicators to results"""
    if " - PASSED" in result:
        return f"✅ {result}"
    elif "_OOM" in result:
        return f"🟡 {result}"
    elif " - FAILED" in result or " - ERROR" in result:
        return f"❌ {result}"
    elif " - SKIPPED" in result:
        return f"⏭️ {result}"
    else:
        return result

def main():
    parser = argparse.ArgumentParser(description='Extract simple test results from pytest logs with validation and OOM detection')
    parser.add_argument('files', nargs='+', help='Log files to analyze (supports wildcards)')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show validation details')
    parser.add_argument('--validate', action='store_true', help='Validate against pytest collection')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary validation')
    parser.add_argument('--include-skipped', action='store_true', help='Include SKIPPED tests in output')
    parser.add_argument('--with-emojis', action='store_true', help='Add emoji indicators to results')
    parser.add_argument('--oom-details', action='store_true', help='Show detailed memory info for OOM failures')
    
    args = parser.parse_args()
    
    all_files = []
    for pattern in args.files:
        if '*' in pattern or '?' in pattern:
            files = glob.glob(pattern)
        else:
            files = [pattern]
        all_files.extend(files)
    
    if not all_files:
        print("No files found matching the patterns")
        return
    
    all_results = []
    
    for filepath in all_files:
        file_result = process_file(filepath, args.validate, args.include_skipped)
        
        if not file_result:
            continue
            
        results = file_result['results']
        summary_counts = file_result['summary_counts']
        summary_text = file_result['summary_text']
        parsed_counts = file_result['parsed_counts']
        validation = file_result['validation']
        
        if args.summary_only:
            # Show only validation summary
            if len(all_files) > 1:
                all_results.append(f"\n# {filepath}")
            
            if summary_text:
                all_results.append(f"Summary: {summary_text}")
                
                # Format parsed counts display
                parsed_display = dict(parsed_counts)
                if 'oom' in parsed_display:
                    oom_count = parsed_display.pop('oom')
                    parsed_display['OOM'] = oom_count
                
                all_results.append(f"Parsed: {parsed_display} (Total: {sum([v for k, v in parsed_counts.items() if k != 'oom'])})")
                
                if validation.get('counts_match'):
                    all_results.append("✓ Counts match!")
                else:
                    all_results.append("✗ Count mismatch!")
                    if args.include_skipped:
                        all_results.append(f"  Expected: P={summary_counts['passed']} F={summary_counts['failed']} S={summary_counts['skipped']} E={summary_counts['error']}")
                        all_results.append(f"  Found:    P={parsed_counts.get('passed',0)} F={parsed_counts.get('failed',0)} S={parsed_counts.get('skipped',0)} E={parsed_counts.get('error',0)}")
                    else:
                        all_results.append(f"  Expected (excluding skipped): P={summary_counts['passed']} F={summary_counts['failed']} E={summary_counts['error']}")
                        all_results.append(f"  Found:    P={parsed_counts.get('passed',0)} F={parsed_counts.get('failed',0)} E={parsed_counts.get('error',0)}")
                    
                    if parsed_counts.get('oom', 0) > 0:
                        all_results.append(f"  OOM failures: {parsed_counts['oom']} (subset of failed/error)")
            else:
                all_results.append("No summary found in log")
        else:
            # Show full results
            if len(all_files) > 1:
                all_results.append(f"\n# {filepath}")
            
            if results:
                if args.with_emojis:
                    formatted_results = [format_result_with_emoji(result) for result in results]
                    all_results.extend(formatted_results)
                else:
                    all_results.extend(results)
                    
                # Show OOM summary if any OOM failures found
                oom_count = parsed_counts.get('oom', 0)
                if oom_count > 0:
                    all_results.append(f"\n# 🟡 {oom_count} CUDA OOM failure(s) detected in {filepath}")
                    oom_tests = [r for r in results if '_OOM' in r]
                    
                    if args.oom_details:
                        # Show detailed OOM information
                        with open(filepath, 'r') as f:
                            content = f.read()
                        summary_info = parse_short_test_summary(content)
                        
                        for test in oom_tests:
                            test_name = test.split(' - ')[0]
                            all_results.append(f"#   🟡 {test_name}")
                            
                            if test_name in summary_info:
                                oom_details = extract_oom_details(summary_info[test_name])
                                if oom_details:
                                    if 'attempted' in oom_details:
                                        all_results.append(f"#      Attempted: {oom_details['attempted']}")
                                    if 'free' in oom_details:
                                        all_results.append(f"#      Available: {oom_details['free']}")
                                    if 'total' in oom_details:
                                        all_results.append(f"#      Total GPU: {oom_details['total']}")
                                    if 'used' in oom_details:
                                        all_results.append(f"#      In Use: {oom_details['used']}")
                    else:
                        for test in oom_tests:
                            test_name = test.split(' - ')[0]
                            all_results.append(f"#   {test_name}")
            else:
                if args.include_skipped:
                    all_results.append("# No test results found")
                else:
                    all_results.append("# No passed/failed test results found (skipped tests excluded)")
            
            # Add validation info if verbose
            if args.verbose and summary_text:
                all_results.append(f"\n# Validation for {filepath}:")
                all_results.append(f"# Summary: {summary_text}")
                all_results.append(f"# Parsed counts: {dict(parsed_counts)}")
                
                if validation.get('counts_match'):
                    all_results.append("# ✓ Parsed counts match summary")
                else:
                    all_results.append("# ✗ Parsed counts don't match summary")
                
                if 'all_available_tests' in validation:
                    all_results.append(f"# Available tests in suite: {validation['all_available_tests']}")
    
    if not all_results:
        if args.include_skipped:
            print("No test results found in any files.")
        else:
            print("No passed/failed test results found in any files (skipped tests excluded).")
        return
    
    output = '\n'.join(all_results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

if __name__ == '__main__':
    main()
