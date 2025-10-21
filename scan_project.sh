#!/bin/bash
# Adaptive Algo Trading Project Scanner
# Run in your project directory: bash scan_project_inline.sh

cat > scan_trading_project.py << 'SCANNER'
SCANNER

chmod +x scan_trading_project.py

echo "✅ Adaptive scanner created: scan_trading_project.py"
echo ""
echo "🚀 Running adaptive scan..."
python3 scan_trading_project.py
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter

class AdaptiveTradingScanner:
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.results = {
            "project_name": self.project_path.name,
            "structure": {},
            "databases": [],
            "modules": {},
            "strategies": [],
            "models": [],
            "config": {},
            "dependencies": [],
            "entry_points": [],
            "data_flow": {},
            "api_endpoints": [],
            "unknown_important": [],  # Important files we can't classify
            "inferences": {},  # Educated guesses about unknowns
            "import_graph": {},  # Who imports whom
            "centrality_scores": {},  # Module importance by connections
        }
        
        # Importance scoring thresholds
        self.importance_threshold = 15  # Score needed to be flagged as important
        
        # Track all Python files for graph analysis
        self._all_modules = {}
        self._import_graph = defaultdict(set)
        
    def scan(self) -> Dict[str, Any]:
        print(f"🔍 Scanning project: {self.project_path.absolute()}")
        print("🧠 Running adaptive analysis...\n")
        
        # Phase 1: Gather all data
        self._scan_structure()
        self._scan_databases()
        self._scan_python_modules()
        self._scan_dependencies()
        self._scan_config()
        
        # Phase 2: Build relationship graph
        self._build_import_graph()
        self._calculate_centrality()
        
        # Phase 3: Identify patterns
        self._identify_data_flow()
        self._detect_api_endpoints()
        
        # Phase 4: Adaptive discovery - find important unknowns
        self._discover_unknowns()
        self._infer_purposes()
        
        return self.results
    
    def _scan_structure(self):
        """Map directory structure and collect files"""
        structure = {}
        py_files = []
        other_files = []
        
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', 'node_modules', 'venv', '.venv', 
                'env', '.env', 'build', 'dist', '.pytest_cache'
            }]
            
            rel_path = Path(root).relative_to(self.project_path)
            structure[str(rel_path)] = {
                "files": [f for f in files if not f.startswith('.')],
                "dirs": dirs
            }
            
            # Collect Python files
            for f in files:
                full_path = str(Path(root) / f)
                if f.endswith('.py'):
                    py_files.append(full_path)
                elif not f.startswith('.'):
                    other_files.append(full_path)
        
        self.results["structure"] = structure
        self.results["python_files_count"] = len(py_files)
        self._python_files = py_files
        self._other_files = other_files
        print(f"  📁 Found {len(py_files)} Python files")
        print(f"  📄 Found {len(other_files)} other files")
    
    def _scan_databases(self):
        """Extract database schemas"""
        db_files = []
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(('.db', '.sqlite', '.sqlite3')):
                    db_path = Path(root) / file
                    db_info = self._analyze_database(db_path)
                    if db_info:
                        db_files.append(db_info)
        
        self.results["databases"] = db_files
        print(f"  💾 Found {len(db_files)} database(s)")
    
    def _analyze_database(self, db_path: Path) -> Dict[str, Any]:
        """Extract schema from SQLite database"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                schema[table] = [
                    {"name": col[1], "type": col[2], "not_null": bool(col[3]), 
                     "default": col[4], "pk": bool(col[5])}
                    for col in columns
                ]
            
            conn.close()
            
            return {
                "path": str(db_path.relative_to(self.project_path)),
                "size_mb": round(db_path.stat().st_size / (1024 * 1024), 2),
                "tables": list(schema.keys()),
                "schema": schema
            }
        except Exception as e:
            return {"path": str(db_path.relative_to(self.project_path)), "error": str(e)}
    
    def _scan_python_modules(self):
        """Deep analysis of Python files with importance scoring"""
        modules = {}
        strategies = []
        models = []
        entry_points = []
        
        for file_path in self._python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                rel_path = str(Path(file_path).relative_to(self.project_path))
                module_info = self._analyze_python_file(content, rel_path)
                
                # Calculate importance score
                importance = self._calculate_importance(module_info, content, rel_path)
                module_info['importance_score'] = importance
                
                modules[rel_path] = module_info
                self._all_modules[rel_path] = module_info
                
                # Classify by pattern matching
                if self._is_strategy_module(module_info, content):
                    strategies.append(rel_path)
                if self._is_model_module(module_info, content):
                    models.append(rel_path)
                if self._is_entry_point(module_info, content):
                    entry_points.append(rel_path)
                    
            except Exception as e:
                modules[rel_path] = {"error": str(e), "importance_score": 0}
        
        self.results["modules"] = modules
        self.results["strategies"] = strategies
        self.results["models"] = models
        self.results["entry_points"] = entry_points
        print(f"  🎯 Found {len(strategies)} strategy module(s)")
        print(f"  🤖 Found {len(models)} ML model module(s)")
        print(f"  🚀 Found {len(entry_points)} entry point(s)")
    
    def _calculate_importance(self, module_info: Dict, content: str, path: str) -> int:
        """Score a module's importance based on multiple signals"""
        score = 0
        
        # Size signal
        lines = module_info.get('lines', 0)
        if lines > 500:
            score += 10
        elif lines > 200:
            score += 5
        
        # Complexity signal
        num_classes = len(module_info.get('classes', []))
        num_functions = len(module_info.get('functions', []))
        if num_classes > 3 or num_functions > 10:
            score += 8
        elif num_classes > 1 or num_functions > 5:
            score += 4
        
        # Main indicator
        if any(f['name'] == 'main' for f in module_info.get('functions', [])):
            score += 5
        if '__name__' in content and '__main__' in content:
            score += 3
        
        # Data operations
        data_imports = ['sqlite3', 'sqlalchemy', 'psycopg2', 'pymongo', 'redis']
        if any(imp in module_info.get('imports', []) for imp in data_imports):
            score += 6
        
        # External APIs / Network
        api_imports = ['requests', 'aiohttp', 'urllib', 'httpx']
        if any(imp in module_info.get('imports', []) for imp in api_imports):
            score += 6
        
        # Config/Settings indicator
        if 'config' in path.lower() or 'settings' in path.lower():
            score += 5
        
        return score
    
    def _analyze_python_file(self, content: str, path: str) -> Dict[str, Any]:
        """Extract classes, functions, imports, and metadata"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"error": "Syntax error"}
        
        info = {
            "classes": [],
            "functions": [],
            "imports": [],
            "decorators": set(),
            "lines": content.count('\n'),
            "docstring": ast.get_docstring(tree),
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info["classes"].append({
                    "name": node.name,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "bases": [self._get_name(base) for base in node.bases],
                    "decorators": [self._get_name(d) for d in node.decorator_list]
                })
            
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                info["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [self._get_name(d) for d in node.decorator_list]
                })
                info["decorators"].update([self._get_name(d) for d in node.decorator_list])
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    info["imports"].extend([alias.name for alias in node.names])
                else:
                    module = node.module or ""
                    info["imports"].append(module)
        
        info["decorators"] = list(info["decorators"])
        return info
    
    def _get_name(self, node) -> str:
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(type(node).__name__)
    
    def _build_import_graph(self):
        """Build who-imports-whom graph"""
        for module_path, info in self._all_modules.items():
            if 'imports' not in info:
                continue
            
            # Convert imports to potential module paths
            for imp in info['imports']:
                # Try to match import to actual file
                potential_paths = self._resolve_import(imp, module_path)
                for target in potential_paths:
                    if target in self._all_modules:
                        self._import_graph[module_path].add(target)
        
        # Store in results
        self.results["import_graph"] = {
            k: list(v) for k, v in self._import_graph.items()
        }
    
    def _resolve_import(self, import_name: str, from_path: str) -> List[str]:
        """Try to resolve an import statement to actual file paths"""
        candidates = []
        
        # Simple heuristic: look for files matching the import
        import_parts = import_name.split('.')
        
        for module_path in self._all_modules.keys():
            module_name = Path(module_path).stem
            # Match if module name appears in import
            if any(part in module_name for part in import_parts):
                candidates.append(module_path)
        
        return candidates
    
    def _calculate_centrality(self):
        """Calculate module centrality (how connected/important)"""
        centrality = {}
        
        for module in self._all_modules.keys():
            # In-degree: how many modules import this one
            in_degree = sum(1 for imports in self._import_graph.values() 
                          if module in imports)
            
            # Out-degree: how many modules this one imports
            out_degree = len(self._import_graph.get(module, []))
            
            # Centrality score
            centrality[module] = {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "centrality": in_degree * 2 + out_degree  # Weighted toward being imported
            }
        
        self.results["centrality_scores"] = centrality
    
    def _is_strategy_module(self, module_info: Dict, content: str) -> bool:
        """Detect trading strategy modules"""
        keywords = ['strategy', 'signal', 'trade', 'backtest', 'indicator', 'alpha']
        
        for cls in module_info.get('classes', []):
            if any(kw in cls['name'].lower() for kw in keywords):
                return True
        
        for func in module_info.get('functions', []):
            if any(kw in func['name'].lower() for kw in keywords):
                return True
        
        return False
    
    def _is_model_module(self, module_info: Dict, content: str) -> bool:
        """Detect ML model modules"""
        ml_imports = ['sklearn', 'torch', 'tensorflow', 'keras', 'xgboost', 'lightgbm', 'catboost']
        
        for imp in module_info.get('imports', []):
            if any(ml_lib in imp.lower() for ml_lib in ml_imports):
                return True
        
        return False
    
    def _is_entry_point(self, module_info: Dict, content: str) -> bool:
        """Detect entry point files"""
        has_main = any(f['name'] == 'main' for f in module_info.get('functions', []))
        has_name_main = '__name__' in content and '__main__' in content
        has_cli = any('argparse' in imp or 'click' in imp for imp in module_info.get('imports', []))
        
        return has_main or has_name_main or has_cli
    
    def _scan_dependencies(self):
        """Extract dependencies"""
        deps = []
        req_file = self.project_path / 'requirements.txt'
        if req_file.exists():
            with open(req_file, 'r') as f:
                deps.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
        
        pyproject = self.project_path / 'pyproject.toml'
        if pyproject.exists():
            with open(pyproject, 'r') as f:
                content = f.read()
                deps.extend(re.findall(r'"([^"]+)"', content))
        
        self.results["dependencies"] = list(set(deps))
        print(f"  📦 Found {len(deps)} dependencies")
    
    def _scan_config(self):
        """Extract configuration files"""
        config_files = ['.env', 'config.yaml', 'config.json', 'settings.py', '.env.example']
        
        for config_file in config_files:
            path = self.project_path / config_file
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                    
                    if config_file.endswith('.json'):
                        self.results["config"][config_file] = json.loads(content)
                    else:
                        keys = re.findall(r'^([A-Z_]+)=', content, re.MULTILINE)
                        self.results["config"][config_file] = keys
                except Exception as e:
                    self.results["config"][config_file] = f"Error: {str(e)}"
    
    def _identify_data_flow(self):
        """Map data flow between components"""
        flow = {
            "data_sources": [],
            "processing_pipeline": [],
            "storage": [],
            "output": []
        }
        
        for module_path, info in self.results["modules"].items():
            if isinstance(info, dict) and "imports" in info:
                # Data sources
                if any(imp in ['requests', 'aiohttp', 'alpaca', 'yfinance', 'ccxt'] 
                      for imp in info["imports"]):
                    flow["data_sources"].append(module_path)
                
                # Storage
                if any(imp in ['sqlite3', 'sqlalchemy', 'psycopg2'] 
                      for imp in info["imports"]):
                    flow["storage"].append(module_path)
                
                # Processing
                if any(imp in ['pandas', 'numpy'] for imp in info["imports"]):
                    flow["processing_pipeline"].append(module_path)
                
                # Output (plotting, reporting)
                if any(imp in ['matplotlib', 'plotly', 'seaborn'] 
                      for imp in info["imports"]):
                    flow["output"].append(module_path)
        
        self.results["data_flow"] = flow
    
    def _detect_api_endpoints(self):
        """Detect API/web endpoints"""
        endpoints = []
        
        for module_path, info in self.results["modules"].items():
            if isinstance(info, dict):
                # Check for web framework imports
                web_frameworks = ['flask', 'fastapi', 'django', 'aiohttp', 'tornado']
                if any(fw in info.get('imports', []) for fw in web_frameworks):
                    endpoints.append({
                        "module": module_path,
                        "type": "web_api",
                        "decorators": info.get('decorators', [])
                    })
        
        self.results["api_endpoints"] = endpoints
        if endpoints:
            print(f"  🌐 Found {len(endpoints)} API module(s)")
    
    def _discover_unknowns(self):
        """Find important files that don't fit known patterns"""
        print("\n🔬 Discovering unknown important components...")
        
        classified = set(
            self.results['strategies'] + 
            self.results['models'] + 
            self.results['entry_points']
        )
        
        unknowns = []
        
        for module_path, info in self.results["modules"].items():
            if module_path in classified:
                continue
            
            if isinstance(info, dict):
                # Check multiple importance signals
                importance_score = info.get('importance_score', 0)
                centrality = self.results["centrality_scores"].get(module_path, {})
                centrality_score = centrality.get('centrality', 0)
                
                total_score = importance_score + centrality_score * 2
                
                if total_score >= self.importance_threshold:
                    unknowns.append({
                        "path": module_path,
                        "importance_score": importance_score,
                        "centrality_score": centrality_score,
                        "total_score": total_score,
                        "classes": len(info.get('classes', [])),
                        "functions": len(info.get('functions', [])),
                        "lines": info.get('lines', 0),
                        "imports": info.get('imports', [])[:10],  # First 10
                    })
        
        # Sort by total score
        unknowns.sort(key=lambda x: x['total_score'], reverse=True)
        
        self.results["unknown_important"] = unknowns
        print(f"  ❓ Found {len(unknowns)} important unclassified modules")
    
    def _infer_purposes(self):
        """Make educated guesses about what unknown components do"""
        print("\n💡 Inferring purposes of unknowns...\n")
        
        inferences = {}
        
        for unknown in self.results["unknown_important"]:
            path = unknown["path"]
            module_info = self.results["modules"].get(path, {})
            
            guesses = []
            confidence = 0
            
            # Analyze imports
            imports = module_info.get('imports', [])
            
            # Inference 1: Configuration/Settings
            if 'config' in path.lower() or 'settings' in path.lower():
                guesses.append("Configuration management")
                confidence += 30
            
            # Inference 2: Data processing utility
            if 'pandas' in imports or 'numpy' in imports:
                guesses.append("Data processing utility")
                confidence += 25
            
            # Inference 3: API client/wrapper
            if any(imp in imports for imp in ['requests', 'aiohttp', 'urllib']):
                guesses.append("API client or data fetcher")
                confidence += 25
            
            # Inference 4: Database operations
            if any(imp in imports for imp in ['sqlite3', 'sqlalchemy']):
                guesses.append("Database operations layer")
                confidence += 25
            
            # Inference 5: Utility/Helper
            if 'util' in path.lower() or 'helper' in path.lower():
                guesses.append("Utility/helper functions")
                confidence += 20
            
            # Inference 6: Testing
            if 'test' in path.lower() or 'pytest' in imports:
                guesses.append("Testing module")
                confidence += 20
            
            # Inference 7: Analysis by class names
            for cls in module_info.get('classes', []):
                name_lower = cls['name'].lower()
                if 'manager' in name_lower:
                    guesses.append(f"Management layer ({cls['name']})")
                    confidence += 15
                if 'client' in name_lower:
                    guesses.append(f"Client implementation ({cls['name']})")
                    confidence += 15
                if 'handler' in name_lower:
                    guesses.append(f"Event/data handler ({cls['name']})")
                    confidence += 15
            
            # Inference 8: Centrality-based
            centrality = self.results["centrality_scores"].get(path, {})
            if centrality.get('in_degree', 0) > 3:
                guesses.append("Core utility (heavily imported)")
                confidence += 20
            
            if not guesses:
                guesses = ["Unknown purpose - needs manual inspection"]
                confidence = 0
            
            inferences[path] = {
                "likely_purposes": list(set(guesses)),
                "confidence": min(confidence, 100),
                "reasoning": {
                    "imports": imports[:5],
                    "classes": [c['name'] for c in module_info.get('classes', [])],
                    "centrality": centrality.get('in_degree', 0)
                }
            }
            
            print(f"  ❓ {path}")
            print(f"     → Likely: {', '.join(guesses)}")
            print(f"     → Confidence: {min(confidence, 100)}%\n")
        
        self.results["inferences"] = inferences
    
    def generate_report(self) -> str:
        """Generate comprehensive human-readable report"""
        report = []
        report.append(f"# 📊 Adaptive Project Analysis: {self.results['project_name']}\n")
        
        # Summary
        report.append("## 📈 Summary")
        report.append(f"- **Python files:** {self.results['python_files_count']}")
        report.append(f"- **Databases:** {len(self.results['databases'])}")
        report.append(f"- **Strategy modules:** {len(self.results['strategies'])}")
        report.append(f"- **ML model modules:** {len(self.results['models'])}")
        report.append(f"- **Entry points:** {len(self.results['entry_points'])}")
        report.append(f"- **API endpoints:** {len(self.results['api_endpoints'])}")
        report.append(f"- **Important unknowns:** {len(self.results['unknown_important'])}")
        report.append(f"- **Dependencies:** {len(self.results['dependencies'])}\n")
        
        # Entry points
        if self.results['entry_points']:
            report.append("## 🚀 Entry Points")
            for ep in self.results['entry_points']:
                report.append(f"- `{ep}`")
            report.append("")
        
        # Databases
        if self.results['databases']:
            report.append("## 💾 Database Schemas")
            for db in self.results['databases']:
                report.append(f"\n### {db['path']}")
                if 'tables' in db:
                    report.append(f"**Size:** {db['size_mb']} MB")
                    report.append(f"**Tables:** {', '.join(db['tables'])}")
                    for table, schema in db.get('schema', {}).items():
                        report.append(f"\n**`{table}`**")
                        for col in schema[:10]:  # Limit columns shown
                            pk = " 🔑" if col['pk'] else ""
                            report.append(f"  - {col['name']} ({col['type']}){pk}")
        
        # Strategies
        if self.results['strategies']:
            report.append("\n## 🎯 Trading Strategies")
            for strategy in self.results['strategies']:
                report.append(f"- `{strategy}`")
                module = self.results['modules'].get(strategy, {})
                if 'classes' in module:
                    for cls in module['classes']:
                        report.append(f"  - Class: `{cls['name']}`")
        
        # Models
        if self.results['models']:
            report.append("\n## 🤖 ML Models")
            for model in self.results['models']:
                report.append(f"- `{model}`")
        
        # API endpoints
        if self.results['api_endpoints']:
            report.append("\n## 🌐 API Endpoints")
            for ep in self.results['api_endpoints']:
                report.append(f"- `{ep['module']}` ({ep['type']})")
        
        # Data flow
        report.append("\n## 🔄 Data Flow")
        flow = self.results.get('data_flow', {})
        if flow.get('data_sources'):
            report.append(f"**Data sources:** {', '.join(f'`{s}`' for s in flow['data_sources'][:5])}")
        if flow.get('processing_pipeline'):
            report.append(f"**Processing:** {', '.join(f'`{p}`' for p in flow['processing_pipeline'][:5])}")
        if flow.get('storage'):
            report.append(f"**Storage:** {', '.join(f'`{s}`' for s in flow['storage'][:5])}")
        if flow.get('output'):
            report.append(f"**Output:** {', '.join(f'`{o}`' for o in flow['output'][:5])}")
        
        # Unknown important modules - THE KEY NEW SECTION
        if self.results['unknown_important']:
            report.append("\n## ❓ Important Unclassified Modules")
            report.append("\n*These modules scored high on importance but don't fit known patterns.*\n")
            
            for unknown in self.results['unknown_important'][:10]:  # Top 10
                path = unknown['path']
                report.append(f"### `{path}` (score: {unknown['total_score']})")
                
                # Add inference if available
                if path in self.results['inferences']:
                    inference = self.results['inferences'][path]
                    report.append(f"**Likely purposes:** {', '.join(inference['likely_purposes'])}")
                    report.append(f"**Confidence:** {inference['confidence']}%")
                
                report.append(f"- Lines: {unknown['lines']}")
                report.append(f"- Classes: {unknown['classes']}, Functions: {unknown['functions']}")
                report.append(f"- Centrality: {unknown['centrality_score']}")
                if unknown['imports']:
                    report.append(f"- Key imports: {', '.join(unknown['imports'][:5])}")
                report.append("")
        
        # Centrality analysis
        report.append("\n## 🕸️ Module Centrality (Most Connected)")
        centrality_sorted = sorted(
            self.results['centrality_scores'].items(),
            key=lambda x: x[1]['centrality'],
            reverse=True
        )[:10]
        
        for module, scores in centrality_sorted:
            if scores['centrality'] > 0:
                report.append(f"- `{module}` (imported by {scores['in_degree']}, imports {scores['out_degree']})")
        
        # Dependencies
        if self.results['dependencies']:
            report.append("\n## 📦 Key Dependencies")
            key_deps = [d for d in self.results['dependencies'] if any(
                kw in d.lower() for kw in ['pandas', 'numpy', 'sklearn', 'torch', 'alpaca', 'sqlite', 'flask', 'fastapi']
            )]
            for dep in key_deps[:15]:
                report.append(f"- {dep}")
        
        return '\n'.join(report)


def main():
    scanner = AdaptiveTradingScanner()
    print("\n" + "="*70)
    results = scanner.scan()
    print("="*70 + "\n")
    
    report = scanner.generate_report()
    print(report)
    
    # Save outputs
    with open('project_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('project_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("✅ Saved: project_analysis.json (structured data)")
    print("✅ Saved: project_report.md (human-readable)")
    print("="*70)


if __name__ == '__main__':
    main()
