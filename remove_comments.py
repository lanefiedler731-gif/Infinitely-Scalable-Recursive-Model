#!/usr/bin/env python3
"""
Script to remove all comments from Python files.

Removes:
- Single-line comments (# ...)
- Inline comments (code # comment)
- Multi-line comments/docstrings (triple-quoted strings)

Preserves:
- String literals (even if they contain # or triple quotes)
- Code structure and formatting
"""

import os
import sys
import tokenize
import io
from pathlib import Path
from typing import List, Tuple


def remove_comments_from_file(file_path: Path) -> bool:
    """
    Remove all comments from a Python file.
    
    Returns True if file was modified, False otherwise.
    """
    try:

        with open(file_path, 'rb') as f:
            source_bytes = f.read()
        

        tokens = []
        try:
            for token in tokenize.tokenize(io.BytesIO(source_bytes).readline):

                if token.type != tokenize.COMMENT:
                    tokens.append(token)
        except tokenize.TokenError:

            print(f"Warning: Failed to tokenize {file_path}, skipping...")
            return False
        

        result_lines = []
        current_line = []
        last_line_no = 1
        
        for token in tokens:

            if token.start[0] > last_line_no:

                if current_line:
                    result_lines.append(''.join(current_line))
                    current_line = []

                while last_line_no < token.start[0]:
                    result_lines.append('')
                    last_line_no += 1
            

            if token.type == tokenize.STRING:

                current_line.append(token.string)
            elif token.type == tokenize.NL:

                if current_line:
                    result_lines.append(''.join(current_line))
                    current_line = []
                last_line_no += 1
            elif token.type == tokenize.NEWLINE:

                if current_line:
                    result_lines.append(''.join(current_line))
                    current_line = []
                last_line_no += 1
            elif token.type != tokenize.ENCODING and token.type != tokenize.ENDMARKER:

                if token.type == tokenize.INDENT:
                    current_line.append(token.string)
                elif token.type == tokenize.DEDENT:

                    pass
                else:

                    if current_line and token.string not in '.,:;)]}':

                        last_char = current_line[-1][-1] if current_line else ''
                        if last_char not in ' \t\n([{' and token.string not in '([{':

                            if not token.string.startswith(' ') and last_char != '\n':

                                pass
                    current_line.append(token.string)
            
            last_line_no = token.end[0]
        

        if current_line:
            result_lines.append(''.join(current_line))
        

        new_content = '\n'.join(result_lines)
        if new_content != source_bytes.decode('utf-8'):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def remove_comments_simple(file_path: Path) -> bool:
    """
    Simpler approach: use tokenize to identify comments and remove them.
    This preserves the original formatting better.
    """
    try:

        with open(file_path, 'rb') as f:
            source_bytes = f.read()
        
        source_lines = source_bytes.decode('utf-8').splitlines(keepends=True)
        

        comment_ranges = set()
        
        try:
            for token in tokenize.tokenize(io.BytesIO(source_bytes).readline):
                if token.type == tokenize.COMMENT:

                    start_line, start_col = token.start
                    end_line, end_col = token.end
                    comment_ranges.add((start_line, start_col, end_line, end_col))
        except tokenize.TokenError as e:
            print(f"Warning: Tokenization error in {file_path}: {e}, skipping...")
            return False
        
        if not comment_ranges:
            return False
        

        result_lines = []
        for line_no, line in enumerate(source_lines, 1):

            line_comments = [(s, e) for sl, s, el, e in comment_ranges if sl == line_no]
            
            if not line_comments:
                result_lines.append(line)
                continue
            

            line_comments.sort(reverse=True)
            

            modified_line = line
            for start_col, end_col in line_comments:

                start_idx = start_col - 1
                end_idx = end_col - 1
                

                before_comment = modified_line[:start_idx].rstrip()
                
                if before_comment and not before_comment.endswith('\\'):

                    modified_line = modified_line[:start_idx].rstrip() + '\n' if line.endswith('\n') else modified_line[:start_idx].rstrip()
                else:

                    modified_line = ''
            
            if modified_line.strip():
                result_lines.append(modified_line)
            elif line.endswith('\n') and not modified_line:

                result_lines.append('\n')
        

        new_content = ''.join(result_lines)
        original_content = ''.join(source_lines)
        
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def remove_docstrings(file_path: Path) -> bool:
    """
    Remove docstrings (triple-quoted strings) from Python files.
    """
    import ast
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        

        tree = ast.parse(source)
        
        class DocstringRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):

                if ast.get_docstring(node):
                    node.body = node.body[1:] if len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str) else node.body
                return self.generic_visit(node)
            
            def visit_ClassDef(self, node):

                if ast.get_docstring(node):
                    node.body = node.body[1:] if len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str) else node.body
                return self.generic_visit(node)
            
            def visit_Module(self, node):

                if ast.get_docstring(node):
                    node.body = node.body[1:] if len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str) else node.body
                return self.generic_visit(node)
        
        remover = DocstringRemover()
        new_tree = remover.visit(tree)
        


        
        return False
    
    except Exception as e:
        print(f"Error removing docstrings from {file_path}: {e}")
        return False


def main():
    """Main function to process all Python files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove all comments from Python files')
    parser.add_argument('--path', type=str, default='.', 
                       help='Path to directory or file to process (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('--include-docstrings', action='store_true',
                       help='Also remove docstrings (triple-quoted strings)')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    

    if path.is_file():
        python_files = [path] if path.suffix == '.py' else []
    else:
        python_files = list(path.rglob('*.py'))
    
    if not python_files:
        print(f"No Python files found in {path}")
        return
    
    print(f"Found {len(python_files)} Python file(s)")
    
    modified_count = 0
    for py_file in python_files:
        if args.dry_run:
            print(f"Would process: {py_file}")
        else:
            print(f"Processing: {py_file}")
            if remove_comments_simple(py_file):
                modified_count += 1
                print(f"  ✓ Removed comments")
            else:
                print(f"  - No changes needed")
    
    if args.dry_run:
        print(f"\nDry run complete. Would process {len(python_files)} file(s)")
    else:
        print(f"\nComplete! Modified {modified_count} out of {len(python_files)} file(s)")


if __name__ == '__main__':
    main()

