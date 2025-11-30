import aiohttp
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def async_query_get(url: str, headers: dict = {}, params: dict = {}) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url=url, headers=headers, params=params) as response:
            print(response.headers)
            data = await response.json()
            return data

async def async_query_post(url: str, json_data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Асинхронный POST запрос с обработкой ошибок
    """
    headers = headers or {"Content-Type": "application/json"}
    json_data = json_data or {}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=url,
                json=json_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"HTTP Error {response.status}: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
    except aiohttp.ClientError as e:
        logger.error(f"Client error for POST {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error for POST {url}: {e}")
        raise