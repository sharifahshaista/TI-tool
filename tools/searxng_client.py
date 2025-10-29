from pydantic_ai import RunContext
import httpx


class SearxNGClient:
    def __init__(self, base_url: str, num_results: int = 10):
        """Configure a SearxNG-backed web search tool."""
        self.base_url = base_url.rstrip("/")
        self.num_results = num_results

    async def search_web(self, ctx: RunContext[None], query: str) -> str:
        """Search the web using SearXNG and summarize the results."""
        return await self._search(query)

    async def _search(self, query: str) -> str:
        search_url = f"{self.base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "number_of_results": self.num_results,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # pragma: no cover - network errors
            return f"Search failed: {exc}"

        results = data.get("results", [])
        if not results:
            return f"No results for '{query}'."

        lines = [f"Search results for '{query}':", ""]
        for index, item in enumerate(results[: self.num_results], start=1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "No description")
            raw_content = item.get("raw_content", "No raw content")
            lines.extend(
                [
                    f"{index}. {title}",
                    f"   URL: {url}",
                    f"   {content}",
                    f"   {raw_content}\n",
                ]
            )
        
        # print("\n".join(lines).rstrip())

        return "\n".join(lines).rstrip()

    def get_tool(self):
        """Return the search function for agent tool registration."""
        return self.search_web

if __name__ == "__main__":
    import asyncio

    SEARXNG_URL = "http://localhost:32768"
    searxng_client = SearxNGClient(SEARXNG_URL, num_results=20)
    searxng_web_tool = searxng_client.get_tool()
    
    # Run the async function using asyncio.run()
    result = asyncio.run(
        searxng_web_tool(None, "What is current and projected Singapore Interest Rate?")
    )
    print(type(result))
    
    print(result)