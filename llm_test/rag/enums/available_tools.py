# coding=utf-8
from llm_test.rag.tools.weather_tool import get_weather
from llm_test.rag.tools.search_attraction import get_attraction


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
