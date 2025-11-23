#!/usr/bin/env python3
"""
Gemini Repository Review Script
Performs a comprehensive code review of the UltraThink Pilot repository using Google's Gemini API.
"""

import os
import sys
import json
import pathlib
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiRepoReviewer:
    """Comprehensive repository reviewer using Gemini API."""

    # File extensions to analyze
    CODE_EXTENSIONS = {'.py', '.yaml', '.yml', '.sh', '.md', '.txt', '.json', '.toml'}

    # Directories to skip
    SKIP_DIRS = {
        '__pycache__', '.git', '.venv', 'venv', 'env',
        'node_modules', '.pytest_cache', '.mypy_cache',
        'dist', 'build', '*.egg-info', '.ipynb_checkpoints'
    }

    # Files to skip
    SKIP_FILES = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
        '.DS_Store', 'Thumbs.db', '.gitignore'
    }

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-thinking-exp-01-21"):
        """
        Initialize Gemini reviewer.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Gemini model to use (default: gemini-2.0-flash-thinking-exp-01-21 for deep analysis)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "REPLACE_WITH_YOUR_GEMINI_API_KEY":
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY in .env file.\n"
                "Get your key from: https://aistudio.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

        # Statistics
        self.stats = {
            'files_analyzed': 0,
            'total_lines': 0,
            'total_chars': 0,
            'api_calls': 0
        }

    def scan_repository(self, repo_path: str = ".") -> Dict[str, List[str]]:
        """
        Scan repository and categorize files.

        Returns:
            Dictionary mapping file categories to file paths
        """
        repo_path = pathlib.Path(repo_path).resolve()
        categorized_files = {
            'agents': [],
            'backtesting': [],
            'rl': [],
            'ml_persistence': [],
            'orchestration': [],
            'tests': [],
            'config': [],
            'docs': [],
            'scripts': [],
            'other': []
        }

        for file_path in repo_path.rglob("*"):
            # Skip directories and excluded patterns
            if file_path.is_dir():
                continue

            # Check if in skip directory
            if any(skip in file_path.parts for skip in self.SKIP_DIRS):
                continue

            # Check file extension
            if file_path.suffix not in self.CODE_EXTENSIONS:
                continue

            # Categorize file
            relative_path = str(file_path.relative_to(repo_path))

            if 'agents/' in relative_path:
                categorized_files['agents'].append(relative_path)
            elif 'backtesting/' in relative_path:
                categorized_files['backtesting'].append(relative_path)
            elif 'rl/' in relative_path:
                categorized_files['rl'].append(relative_path)
            elif 'ml_persistence/' in relative_path:
                categorized_files['ml_persistence'].append(relative_path)
            elif 'orchestration/' in relative_path:
                categorized_files['orchestration'].append(relative_path)
            elif 'test' in relative_path.lower():
                categorized_files['tests'].append(relative_path)
            elif file_path.suffix in {'.yaml', '.yml', '.json', '.toml'}:
                categorized_files['config'].append(relative_path)
            elif file_path.suffix == '.md':
                categorized_files['docs'].append(relative_path)
            elif file_path.suffix == '.sh' or file_path.name.startswith('run_'):
                categorized_files['scripts'].append(relative_path)
            else:
                categorized_files['other'].append(relative_path)

        return categorized_files

    def read_file_content(self, file_path: str, max_lines: int = 1000) -> str:
        """Read file content with line limits."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                self.stats['total_lines'] += len(lines)

                if len(lines) > max_lines:
                    content = ''.join(lines[:max_lines]) + f"\n\n... (truncated {len(lines) - max_lines} lines)"
                else:
                    content = ''.join(lines)

                self.stats['total_chars'] += len(content)
                return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def create_code_summary(self, categorized_files: Dict[str, List[str]], repo_path: str = ".") -> str:
        """Create a structured summary of repository code."""
        summary_parts = []
        summary_parts.append("# UltraThink Pilot Repository Code Summary\n")
        summary_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for category, files in categorized_files.items():
            if not files:
                continue

            summary_parts.append(f"\n## {category.upper()} ({len(files)} files)\n\n")

            for file_path in sorted(files)[:20]:  # Limit files per category
                full_path = os.path.join(repo_path, file_path)
                summary_parts.append(f"### File: {file_path}\n")
                summary_parts.append("```")

                # Add file extension for syntax highlighting
                ext = pathlib.Path(file_path).suffix
                if ext == '.py':
                    summary_parts.append("python")
                elif ext in {'.yaml', '.yml'}:
                    summary_parts.append("yaml")
                elif ext == '.sh':
                    summary_parts.append("bash")
                elif ext == '.json':
                    summary_parts.append("json")

                summary_parts.append(f"\n{self.read_file_content(full_path)}\n```\n\n")
                self.stats['files_analyzed'] += 1

        return '\n'.join(summary_parts)

    def review_with_gemini(self, content: str, review_type: str = "comprehensive") -> str:
        """
        Send content to Gemini for review.

        Args:
            content: Code content to review
            review_type: Type of review (comprehensive, architecture, security, performance)
        """
        prompts = {
            "comprehensive": """You are an expert software architect and code reviewer. Perform a comprehensive review of this cryptocurrency trading system repository.

Analyze the following aspects:

1. **Architecture & Design**
   - System architecture and component interactions
   - Design patterns and their appropriateness
   - Separation of concerns and modularity
   - Scalability and extensibility

2. **Code Quality**
   - Code organization and structure
   - Naming conventions and readability
   - Code duplication and reusability
   - Error handling and edge cases

3. **Trading System Specific**
   - Agent design and coordination
   - Risk management implementation
   - Backtesting framework robustness
   - RL system architecture and training

4. **Testing & Reliability**
   - Test coverage and quality
   - Integration testing approach
   - Mock/fixture design
   - CI/CD considerations

5. **Performance & Optimization**
   - Algorithmic efficiency
   - Resource utilization
   - Bottleneck identification
   - CUDA/GPU optimization

6. **Security & Best Practices**
   - API key management
   - Input validation
   - Error exposure
   - Dependency security

7. **Documentation & Maintainability**
   - Code documentation quality
   - README and setup instructions
   - Configuration management
   - Onboarding ease

For each section, provide:
- ✅ Strengths (what's done well)
- ⚠️ Issues (problems found)
- 💡 Recommendations (specific improvements)
- 🎯 Priority (High/Medium/Low)

Be specific with file names and line references where applicable.""",

            "architecture": """Perform a deep architectural analysis of this trading system. Focus on:
- Component relationships and data flow
- Agent communication patterns
- State management
- Extension points and flexibility
- Technical debt and architectural smells""",

            "security": """Conduct a security audit focusing on:
- Credential and API key handling
- Input validation and sanitization
- Error information leakage
- Dependency vulnerabilities
- Trade execution safety""",

            "performance": """Analyze performance and optimization:
- Algorithm complexity
- Memory usage patterns
- I/O bottlenecks
- Parallelization opportunities
- GPU/CUDA utilization"""
        }

        prompt = prompts.get(review_type, prompts["comprehensive"])
        full_prompt = f"{prompt}\n\n---\n\n{content}"

        try:
            self.stats['api_calls'] += 1
            print(f"🔄 Calling Gemini API (call #{self.stats['api_calls']})...")

            response = self.model.generate_content(full_prompt)
            return response.text

        except Exception as e:
            return f"Error during Gemini API call: {str(e)}\n\nPlease check your API key and quota."

    def generate_structure_overview(self, categorized_files: Dict[str, List[str]]) -> str:
        """Generate repository structure overview."""
        overview = ["# Repository Structure Overview\n"]

        total_files = sum(len(files) for files in categorized_files.values())
        overview.append(f"**Total Files Analyzed:** {total_files}\n")

        for category, files in sorted(categorized_files.items()):
            if files:
                overview.append(f"\n## {category.title()} ({len(files)} files)")
                for f in sorted(files):
                    overview.append(f"- {f}")

        return '\n'.join(overview)

    def run_full_review(self, repo_path: str = ".", output_file: str = None) -> Dict[str, Any]:
        """
        Run complete repository review.

        Args:
            repo_path: Path to repository
            output_file: Optional output file path

        Returns:
            Dictionary with review results
        """
        print("🚀 Starting Gemini Repository Review...")
        print(f"📁 Repository: {os.path.abspath(repo_path)}")
        print(f"🤖 Model: {self.model_name}\n")

        # Step 1: Scan repository
        print("📊 Scanning repository structure...")
        categorized_files = self.scan_repository(repo_path)
        structure_overview = self.generate_structure_overview(categorized_files)
        print(f"✓ Found {sum(len(f) for f in categorized_files.values())} relevant files\n")

        # Step 2: Create code summary
        print("📝 Creating code summary...")
        code_summary = self.create_code_summary(categorized_files, repo_path)
        print(f"✓ Analyzed {self.stats['files_analyzed']} files ({self.stats['total_lines']} lines)\n")

        # Step 3: Send to Gemini for review
        print("🧠 Sending to Gemini for comprehensive review...")
        print("   (This may take a few minutes for deep analysis...)\n")

        review_content = f"{structure_overview}\n\n{code_summary}"
        gemini_review = self.review_with_gemini(review_content, "comprehensive")

        # Step 4: Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'statistics': self.stats,
            'structure_overview': structure_overview,
            'gemini_review': gemini_review
        }

        # Step 5: Save results
        if output_file is None:
            output_file = f"gemini_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        self.save_review(results, output_file)

        print(f"\n✅ Review complete!")
        print(f"📄 Report saved to: {output_file}")
        print(f"\n📈 Statistics:")
        print(f"   - Files analyzed: {self.stats['files_analyzed']}")
        print(f"   - Total lines: {self.stats['total_lines']:,}")
        print(f"   - API calls: {self.stats['api_calls']}")

        return results

    def save_review(self, results: Dict[str, Any], output_file: str):
        """Save review results to markdown file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Gemini Repository Review - UltraThink Pilot\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Model:** {results['model']}\n")
            f.write(f"**Files Analyzed:** {results['statistics']['files_analyzed']}\n")
            f.write(f"**Total Lines:** {results['statistics']['total_lines']:,}\n\n")
            f.write("---\n\n")
            f.write(results['structure_overview'])
            f.write("\n\n---\n\n")
            f.write("# Comprehensive Review\n\n")
            f.write(results['gemini_review'])
            f.write("\n\n---\n\n")
            f.write(f"*Review completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Gemini Repository Review")
    parser.add_argument(
        '--repo-path',
        default='.',
        help='Path to repository (default: current directory)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--model',
        default='gemini-2.0-flash-thinking-exp-01-21',
        help='Gemini model to use (default: gemini-2.0-flash-thinking-exp-01-21)'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (overrides .env)'
    )

    args = parser.parse_args()

    try:
        reviewer = GeminiRepoReviewer(api_key=args.api_key, model=args.model)
        reviewer.run_full_review(
            repo_path=args.repo_path,
            output_file=args.output
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
