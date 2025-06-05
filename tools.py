"""
Tool implementations for Enhanced R-Search Framework.
"""

import logging
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for tools."""
    
    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the tool with given query and context."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get tool description for agent selection."""
        pass


class SearchEngine(ABC):
    """Abstract base class for search engines."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Search for documents given a query."""
        pass


class MockSearchEngine(SearchEngine):
    """Enhanced mock search engine."""
    
    def __init__(self):
        self.mock_corpus = [
            {"title": "Bank of America", "content": "In 2004, Bank of America announced it would purchase Boston-based bank FleetBoston Financial for $47 billion in cash and stock."},
            {"title": "Bank of America Home Loans", "content": "On July 1, 2008, Bank of America Corporation completed its purchase of Countrywide Financial Corporation."},
            {"title": "Cheryl Dunye", "content": "Cheryl Dunye (born May 13, 1966) is a Liberian-American film director, producer, screenwriter, editor and actress."},
            {"title": "Eric Rohmer", "content": "Jean Marie Maurice Scherer, known as Eric Rohmer (21 March 1920 – 11 January 2010), was a French film director."},
            {"title": "My Baby's Daddy", "content": "My Baby's Daddy is a 2004 American comedy film, directed by Cheryl Dunye."},
            {"title": "A Tale of Winter", "content": "A Tale of Winter is a 1992 French drama film directed by Eric Rohmer."}
        ]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Enhanced search with better relevance."""
        query_lower = query.lower()
        results = []
        
        for doc in self.mock_corpus:
            score = 0
            for word in query_lower.split():
                if word in doc["content"].lower():
                    score += 2
                if word in doc["title"].lower():
                    score += 3
            
            if score > 0:
                results.append((score, doc))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:top_k]]


class SearchTool(Tool):
    """Enhanced search tool with multiple backends."""
    
    def __init__(self, search_engines: List[SearchEngine]):
        self.search_engines = search_engines
        self.query_history = []
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute search across multiple engines."""
        all_results = []
        for engine in self.search_engines:
            try:
                results = engine.search(query, top_k=5)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search engine failed: {e}")
        
        # Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        self.query_history.append(query)
        
        return {
            "results": unique_results[:10],
            "query": query,
            "total_sources": len(unique_results)
        }
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = hash(result.get('content', '')[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def get_description(self) -> str:
        return "Search for information across multiple sources and databases"


class CalculatorTool(Tool):
    """Mathematical calculation tool."""
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute mathematical calculations."""
        try:
            # Simple expression evaluation (in production, use safer evaluation)
            result = eval(query.replace("^", "**"))
            return {
                "result": result,
                "expression": query,
                "success": True
            }
        except Exception as e:
            return {
                "result": None,
                "expression": query,
                "success": False,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        return "Perform mathematical calculations and solve equations"


class KnowledgeGraphTool(Tool):
    """Knowledge graph reasoning tool."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_sample_graph()
    
    def _build_sample_graph(self):
        """Build a sample knowledge graph."""
        # Sample entities and relations
        entities = [
            ("Bank of America", "Corporation"),
            ("FleetBoston Financial", "Corporation"),
            ("Countrywide Financial", "Corporation"),
            ("Cheryl Dunye", "Person"),
            ("Eric Rohmer", "Person"),
            ("My Baby's Daddy", "Film"),
            ("A Tale of Winter", "Film")
        ]
        
        relations = [
            ("Bank of America", "acquired", "FleetBoston Financial", {"year": 2004}),
            ("Bank of America", "acquired", "Countrywide Financial", {"year": 2008}),
            ("Cheryl Dunye", "directed", "My Baby's Daddy", {"year": 2004}),
            ("Eric Rohmer", "directed", "A Tale of Winter", {"year": 1992}),
            ("Cheryl Dunye", "born", "1966", {}),
            ("Eric Rohmer", "born", "1920", {})
        ]
        
        for entity, entity_type in entities:
            self.graph.add_node(entity, type=entity_type)
        
        for subj, rel, obj, attrs in relations:
            self.graph.add_edge(subj, obj, relation=rel, **attrs)
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the knowledge graph."""
        results = []
        query_lower = query.lower()
        
        # Simple keyword-based graph search
        for node in self.graph.nodes():
            if any(word in node.lower() for word in query_lower.split()):
                # Get connected nodes and relations
                connections = []
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph[node][neighbor]
                    connections.append({
                        "target": neighbor,
                        "relation": edge_data.get("relation", "related"),
                        "attributes": {k: v for k, v in edge_data.items() if k != "relation"}
                    })
                
                results.append({
                    "entity": node,
                    "type": self.graph.nodes[node].get("type", "unknown"),
                    "connections": connections
                })
        
        return {
            "results": results,
            "query": query,
            "graph_size": len(self.graph.nodes())
        }
    
    def get_description(self) -> str:
        return "Query structured knowledge graphs for entity relationships"


class CodeExecutorTool(Tool):
    """Code execution tool for programming tasks."""
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code safely (simplified implementation)."""
        try:
            # Simple Python expression evaluation
            # In production, use a sandboxed environment
            if query.strip().startswith("print("):
                # Handle print statements
                result = eval(query.strip()[6:-1])  # Remove print() wrapper
                return {
                    "result": result,
                    "output": str(result),
                    "success": True,
                    "code": query
                }
            else:
                result = eval(query)
                return {
                    "result": result,
                    "output": str(result),
                    "success": True,
                    "code": query
                }
        except Exception as e:
            return {
                "result": None,
                "output": "",
                "success": False,
                "error": str(e),
                "code": query
            }
    
    def get_description(self) -> str:
        return "Execute simple Python code and mathematical expressions"


class WebScraperTool(Tool):
    """Web scraping tool for real-time information."""
    
    def __init__(self):
        # Mock data for demonstration
        self.mock_web_data = {
            "weather": "Current temperature: 22°C, Sunny",
            "news": "Breaking: Technology advances in AI research continue to accelerate",
            "stock": "Market data: AAPL $150.25 (+2.3%)",
        }
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Scrape web content (mocked implementation)."""
        query_lower = query.lower()
        
        # Simple keyword matching for demo
        if "weather" in query_lower:
            content = self.mock_web_data["weather"]
        elif "news" in query_lower:
            content = self.mock_web_data["news"]
        elif "stock" in query_lower or "market" in query_lower:
            content = self.mock_web_data["stock"]
        else:
            content = "No relevant web content found for this query."
        
        return {
            "content": content,
            "query": query,
            "source": "mock_website",
            "timestamp": "2025-01-01T12:00:00Z",
            "success": True
        }
    
    def get_description(self) -> str:
        return "Scrape real-time information from websites"


def create_default_tools() -> Dict[str, Tool]:
    """Create a set of default tools for the framework."""
    search_engine = MockSearchEngine()
    
    return {
        "search": SearchTool([search_engine]),
        "calculator": CalculatorTool(),
        "knowledge_graph": KnowledgeGraphTool(),
        "code_executor": CodeExecutorTool(),
        "web_scraper": WebScraperTool()
    }