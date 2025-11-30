import aiohttp
import asyncio
import json
import random
from typing import List, Dict, Any
import time

class FNSAsyncChecker:
    def __init__(self, base_url: str, api_key: str, inns: List[str], output_file: str = "fns_results.json"):
        self.base_url = base_url
        self.api_key = api_key
        self.inns = inns
        self.output_file = output_file
        self.results = []
        
    async def fetch_inn_data(self, session: aiohttp.ClientSession, inn: str) -> Dict[str, Any]:
        """Выполняет один запрос для конкретного ИНН"""
        url = f"{self.base_url}/api/check"
        params = {
            "key": self.api_key,
            "req": inn
        }
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "inn": inn,
                        "status": "success",
                        "data": data,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "inn": inn,
                        "status": f"error_{response.status}",
                        "data": None,
                        "timestamp": time.time()
                    }
                    
        except asyncio.TimeoutError:
            return {
                "inn": inn,
                "status": "timeout",
                "data": None,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "inn": inn,
                "status": f"exception_{str(e)}",
                "data": None,
                "timestamp": time.time()
            }
    
    async def process_batch(self, inns_batch: List[str]) -> List[Dict[str, Any]]:
        """Обрабатывает батч ИННов"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)  # Ограничение параллельных соединений
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for inn in inns_batch:
                task = self.fetch_inn_data(session, inn)
                tasks.append(task)
                
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем исключения
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "inn": "unknown",
                        "status": f"gather_exception_{str(result)}",
                        "data": None,
                        "timestamp": time.time()
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def save_results(self):
        """Сохраняет результаты в файл"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в {self.output_file}")
    
    async def run_all_requests(self, batch_size: int = 20):
        """Выполняет все 80 запросов асинхронно"""
        print(f"Начинаем обработку {len(self.inns)} ИННов...")
        start_time = time.time()
        
        # Разбиваем на батчи чтобы не перегружать сервер
        batches = [self.inns[i:i + batch_size] for i in range(0, len(self.inns), batch_size)]
        
        for i, batch in enumerate(batches):
            print(f"Обрабатываю батч {i+1}/{len(batches)} ({len(batch)} ИННов)")
            
            batch_results = await self.process_batch(batch)
            self.results.extend(batch_results)
            
            # Сохраняем после каждого батча на случай ошибок
            self.save_results()
            
            # Небольшая пауза между батчами
            if i < len(batches) - 1:
                await asyncio.sleep(1)
        
        end_time = time.time()
        print(f"Обработка завершена за {end_time - start_time:.2f} секунд")
        
        # Статистика
        successful = sum(1 for r in self.results if r['status'] == 'success')
        print(f"Успешных запросов: {successful}/{len(self.inns)}")

def generate_random_inns(count: int = 80) -> List[str]:
    """Генерирует список случайных действующих ИНН российских компаний"""
    # Это примерные ИНН известных российских компаний
    # В реальном использовании нужно брать из вашей базы данных
    sample_inns = [
        "7707083893",  # Сбербанк
        "7736050003",  # Газпром
        "7707033437",  # Роснефть
        "7744001497",  # Яндекс
        "7702070139",  # Лукойл
        "7830002293",  # ВТБ
        "7710030411",  # РЖД
        "7708514824",  # МТС
        "7718249396",  # Мегафон
        "1660048363",  # Татнефть
        "2315117051",  # Новороссийский морской торговый порт
        "2721135590",  # Аэрофлот
        "7702204400",  # Северсталь
        "3664069397",  # НЛМК
        "1653003353",  # Камаз
        "6312101021",  # АвтоВАЗ
        "7726575524",  # Магнит
        "2310021996",  # Ростелеком
        "7706092528",  # X5 Retail Group
        "5024054155",  # МВидео
    ]
    
    # Если нужно больше ИННов, чем есть в примере, дублируем и модифицируем
    if count > len(sample_inns):
        result_inns = sample_inns.copy()
        # Генерируем дополнительные ИННы
        while len(result_inns) < count:
            # Берем случайный ИНН из списка и немного меняем
            base_inn = random.choice(sample_inns)
            new_inn = base_inn[:-2] + str(random.randint(10, 99))
            if new_inn not in result_inns:
                result_inns.append(new_inn)
        return result_inns[:count]
    else:
        return sample_inns[:count]

async def main():
    # Конфигурация
    FNS_URL = "https://api-fns.ru"  # Замените на реальный URL
    API_KEY = "1e5753f4150eb036e83f9d8f025c6fe4659d6e61"   # Замените на ваш API ключ
    OUTPUT_FILE = "fns_check_results.json"
    NUM_REQUESTS = 80
    
    # Генерируем ИННы (в реальном сценарии берите из вашей БД)
    inns_list = generate_random_inns(NUM_REQUESTS)
    
    print(f"Сгенерировано {len(inns_list)} ИНН для проверки")
    print(f"Первые 5 ИНН: {inns_list[:5]}")
    
    # Создаем и запускаем проверку
    checker = FNSAsyncChecker(FNS_URL, API_KEY, inns_list, OUTPUT_FILE)
    await checker.run_all_requests(batch_size=20)

if __name__ == "__main__":
    # Для Windows может потребоваться другой event loop policy
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())