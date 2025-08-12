"""
Singleton Design Pattern - Návrhový vzor Singleton (Optimized Thread-Safe Version)

Singleton je kreativní návrhový vzor, který zajišťuje, že třída má pouze jednu instanci
a poskytuje globální přístupový bod k této instanci.

Tento vzor je užitečný v situacích, kdy potřebujeme:
- Pouze jednu instanci třídy (např. databázové připojení, logger, konfigurace)
- Globální přístup k této instanci
- Líné vytvoření instance (lazy initialization)
- Thread-safe přístup v multi-thread prostředí

Vylepšení v této verzi:
- Thread-safe implementace pomocí threading.Lock
- Optimalizace paměti pomocí __slots__
- Reset() metody pro testování
- Pokročilé funkce pro správu konfigurace a připojení
"""

import threading
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

# YAML je volitelné - pokud není dostupné, používáme pouze JSON
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class DatabaseConnection:
    """
    Thread-safe Singleton třída pro databázové připojení.
    Zajišťuje, že existuje pouze jedno připojení k databázi.
    
    Vylepšení:
    - Thread-safe implementace s threading.Lock
    - Podpora různých connection stringů
    - Reset() metoda pro testování
    - Efektivnější kontrola inicializace
    """
    
    __slots__ = ['connection_string', 'is_connected', '_initialized']
    
    _instance: Optional['DatabaseConnection'] = None
    _lock = threading.Lock()  # Thread-safe lock
    
    def __new__(cls, connection_string: str = "sqlite:///example.db"):
        """
        Thread-safe vytvoření singleton instance.
        """
        with cls._lock:  # Zajišťuje thread-safety
            if cls._instance is None:
                print(f"Vytvářím novou instanci DatabaseConnection")
                cls._instance = super(DatabaseConnection, cls).__new__(cls)
                cls._instance._initialized = False
            else:
                print(f"Vracím existující instanci DatabaseConnection")
        return cls._instance
    
    def __init__(self, connection_string: str = "sqlite:///example.db"):
        """
        Inicializace instance - volá se pouze jednou díky _initialized flagu.
        Nyní podporuje různé connection stringy.
        """
        if not self._initialized:
            self.connection_string = connection_string
            self.is_connected = False
            self._initialized = True
            print(f"DatabaseConnection inicializováno s: {self.connection_string}")
    
    def connect(self) -> bool:
        """
        Thread-safe připojení k databázi.
        
        Returns:
            bool: True pokud se připojení podařilo, False pokud už bylo připojeno
        """
        with DatabaseConnection._lock:
            if not self.is_connected:
                self.is_connected = True
                print("Připojeno k databázi")
                return True
            else:
                print("Již připojeno k databázi")
                return False
    
    def disconnect(self) -> bool:
        """
        Thread-safe odpojení od databáze.
        
        Returns:
            bool: True pokud se odpojení podařilo, False pokud už bylo odpojeno
        """
        with DatabaseConnection._lock:
            if self.is_connected:
                self.is_connected = False
                print("Odpojeno od databáze")
                return True
            else:
                print("Není připojeno k databázi")
                return False
    
    def get_status(self) -> str:
        """Vrací stav připojení (thread-safe)"""
        with DatabaseConnection._lock:
            return f"Připojení: {'aktivní' if self.is_connected else 'neaktivní'}"
    
    def update_connection_string(self, new_connection_string: str) -> None:
        """
        Aktualizuje connection string (vyžaduje odpojení).
        
        Args:
            new_connection_string: Nový connection string
            
        Raises:
            RuntimeError: Pokud je databáze stále připojená
        """
        with DatabaseConnection._lock:
            if self.is_connected:
                raise RuntimeError("Nelze změnit connection string při aktivním připojení")
            self.connection_string = new_connection_string
            print(f"Connection string změněn na: {new_connection_string}")
    
    @classmethod
    def reset(cls) -> None:
        """
        Resetuje singleton instanci - užitečné pro testování.
        POZOR: Použijte pouze v testech!
        """
        with cls._lock:
            if cls._instance is not None:
                if cls._instance.is_connected:
                    cls._instance.disconnect()
                cls._instance = None
                print("DatabaseConnection singleton resetován")


class ConfigManager:
    """
    Thread-safe Singleton třída pro správu konfigurace aplikace.
    
    Vylepšení:
    - Thread-safe implementace
    - Optimalizace paměti pomocí __slots__
    - Načítání z JSON/YAML souborů
    - Reset() metoda pro testování
    """
    
    __slots__ = ['_config', '_initialized']  # Optimalizace paměti
    
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()  # Thread-safe lock - správně inicializován
    
    def __new__(cls):
        """Thread-safe vytvoření singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
                print("Vytvořen nový ConfigManager")
        return cls._instance
    
    def __init__(self):
        """Inicializace - volá se pouze jednou."""
        if not self._initialized:
            self._config: Dict[str, Any] = {}
            self._initialized = True
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Thread-safe nastavení konfigurační hodnoty.
        
        Args:
            key: Klíč konfigurace
            value: Hodnota konfigurace
        """
        with ConfigManager._lock:
            self._config[key] = value
            print(f"Nastavena konfigurace: {key} = {value}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Thread-safe získání konfigurační hodnoty.
        
        Args:
            key: Klíč konfigurace
            default: Výchozí hodnota pokud klíč neexistuje
            
        Returns:
            Hodnota konfigurace nebo default
        """
        with ConfigManager._lock:
            return self._config.get(key, default)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Vrací kopii všech konfiguračních hodnot (thread-safe)."""
        with ConfigManager._lock:
            return self._config.copy()
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Načte konfiguraci z JSON nebo YAML souboru.
        
        Args:
            file_path: Cesta k souboru s konfigurací
            
        Returns:
            bool: True pokud se načtení podařilo, False jinak
        """
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"Soubor {file_path} neexistuje")
                return False
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    config_data = yaml.safe_load(f)
                elif path.suffix.lower() in ['.yaml', '.yml'] and not YAML_AVAILABLE:
                    print("YAML podpora není dostupná. Nainstalujte PyYAML.")
                    return False
                else:
                    print(f"Nepodporovaný formát souboru: {path.suffix}")
                    return False
            
            with ConfigManager._lock:
                self._config.update(config_data)
                print(f"Konfigurace načtena ze souboru: {file_path}")
                return True
                
        except Exception as e:
            print(f"Chyba při načítání konfigurace: {e}")
            return False
    
    def save_to_file(self, file_path: str, format_type: str = 'json') -> bool:
        """
        Uloží konfiguraci do souboru.
        
        Args:
            file_path: Cesta k výstupnímu souboru
            format_type: Formát souboru ('json' nebo 'yaml')
            
        Returns:
            bool: True pokud se uložení podařilo, False jinak
        """
        try:
            with ConfigManager._lock:
                config_copy = self._config.copy()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'json':
                    json.dump(config_copy, f, indent=2, ensure_ascii=False)
                elif format_type.lower() == 'yaml' and YAML_AVAILABLE:
                    yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)
                elif format_type.lower() == 'yaml' and not YAML_AVAILABLE:
                    print("YAML podpora není dostupná. Nainstalujte PyYAML.")
                    return False
                else:
                    print(f"Nepodporovaný formát: {format_type}")
                    return False
            
            print(f"Konfigurace uložena do: {file_path}")
            return True
            
        except Exception as e:
            print(f"Chyba při ukládání konfigurace: {e}")
            return False
    
    @classmethod
    def reset(cls) -> None:
        """
        Resetuje singleton instanci - užitečné pro testování.
        POZOR: Použijte pouze v testech!
        """
        with cls._lock:
            cls._instance = None
            print("ConfigManager singleton resetován")


# Thread-safe Metaclass implementace
class SingletonMeta(type):
    """
    Thread-safe metaclass implementace Singleton patternu.
    Toto je pokročilejší způsob implementace.
    """
    
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()  # Thread-safe lock
    
    def __call__(cls, *args, **kwargs):
        """Thread-safe vytvoření instance."""
        with cls._lock:
            if cls not in cls._instances:
                print(f"Vytvářím novou instanci {cls.__name__}")
                cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
            else:
                print(f"Vracím existující instanci {cls.__name__}")
        return cls._instances[cls]
    
    @classmethod
    def reset_instance(mcs, cls: type) -> None:
        """
        Resetuje instanci specifické třídy.
        
        Args:
            cls: Třída jejíž instanci chceme resetovat
        """
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]
                print(f"Instance {cls.__name__} resetována")


class Logger(metaclass=SingletonMeta):
    """
    Thread-safe Singleton Logger třída používající metaclass.
    
    Vylepšení:
    - Thread-safe operace
    - Optimalizace paměti
    - Reset funkce
    """
    
    __slots__ = ['logs', '_lock']  # Optimalizace paměti
    
    def __init__(self):
        self.logs = []
        self._lock = threading.Lock()  # Instance-specific lock pro logs
        print("Logger inicializován")
    
    def log(self, message: str) -> None:
        """
        Thread-safe přidání zprávy do logu.
        
        Args:
            message: Zpráva k zalogování
        """
        with self._lock:
            self.logs.append(f"[{len(self.logs)+1}] {message}")
            print(f"Zalogováno: {message}")
    
    def get_logs(self) -> list:
        """Vrací kopii všech logů (thread-safe)."""
        with self._lock:
            return self.logs.copy()
    
    def clear_logs(self) -> None:
        """Thread-safe vymazání všech logů."""
        with self._lock:
            self.logs.clear()
            print("Logy vymazány")
    
    def get_log_count(self) -> int:
        """Vrací počet logů (thread-safe)."""
        with self._lock:
            return len(self.logs)
    
    @classmethod
    def reset(cls) -> None:
        """Resetuje Logger singleton instanci."""
        SingletonMeta.reset_instance(cls)


def demonstrate_singleton():
    """
    Demonstrace fungování optimalizovaného Singleton patternu.
    """
    print("=== Demonstrace Thread-Safe Singleton Design Pattern ===\n")
    
    # Test DatabaseConnection
    print("1. Test DatabaseConnection:")
    db1 = DatabaseConnection()
    db2 = DatabaseConnection("postgresql://localhost:5432/test")  # Jiný connection string
    
    print(f"db1 a db2 jsou stejný objekt: {db1 is db2}")
    print(f"Connection string: {db1.connection_string}")
    
    db1.connect()
    print(f"Stav db2: {db2.get_status()}")
    
    # Test změny connection stringu
    try:
        db1.update_connection_string("mysql://localhost:3306/test")
    except RuntimeError as e:
        print(f"Očekávaná chyba: {e}")
    
    db1.disconnect()
    db1.update_connection_string("mysql://localhost:3306/test")
    print()
    
    # Test ConfigManager
    print("2. Test ConfigManager:")
    config1 = ConfigManager()
    config2 = ConfigManager()
    
    config1.set_config("debug", True)
    config1.set_config("max_connections", 100)
    config1.set_config("database_url", "sqlite:///app.db")
    
    print(f"config1 a config2 jsou stejný objekt: {config1 is config2}")
    print(f"Konfigurace z config2: {config2.get_all_config()}")
    
    # Test uložení a načtení konfigurace
    if config1.save_to_file("test_config.json"):
        print("Konfigurace uložena do test_config.json")
    print()
    
    # Test Logger (metaclass implementace)
    print("3. Test Logger (metaclass):")
    logger1 = Logger()
    logger2 = Logger()
    
    logger1.log("První zpráva")
    logger1.log("Druhá zpráva")
    logger1.log("Třetí zpráva")
    
    print(f"logger1 a logger2 jsou stejný objekt: {logger1 is logger2}")
    print(f"Počet logů: {logger2.get_log_count()}")
    print(f"Posledních 2 logy: {logger2.get_logs()[-2:]}")
    print()


def demonstrate_multithread():
    """
    Demonstrace thread-safety v multi-thread prostředí.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    print("4. Test Thread-Safety:")
    
    def worker_function(worker_id: int):
        """Pracovní funkce pro demonstraci thread-safety."""
        # Test DatabaseConnection
        db = DatabaseConnection()
        db.connect()
        time.sleep(0.1)  # Simulace práce
        status = db.get_status()
        
        # Test Logger
        logger = Logger()
        logger.log(f"Zpráva od worker {worker_id}")
        
        # Test ConfigManager
        config = ConfigManager()
        config.set_config(f"worker_{worker_id}", f"value_{worker_id}")
        
        return f"Worker {worker_id}: {status}"
    
    # Spuštění více threadů současně
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_function, i) for i in range(5)]
        results = [future.result() for future in futures]
    
    for result in results:
        print(result)
    
    # Ověření výsledků
    logger = Logger()
    config = ConfigManager()
    print(f"Celkový počet logů: {logger.get_log_count()}")
    print(f"Konfigurace workerů: {[k for k in config.get_all_config().keys() if k.startswith('worker_')]}")


def demonstrate_reset_functionality():
    """
    Demonstrace reset() funkcí pro testování.
    """
    print("\n5. Test Reset Funkcí:")
    
    # Vytvoření instancí
    db = DatabaseConnection()
    config = ConfigManager()
    logger = Logger()
    
    # Nastavení nějakých dat
    db.connect()
    config.set_config("test", "value")
    logger.log("Test zpráva")
    
    print(f"Před resetem - DB status: {db.get_status()}")
    print(f"Před resetem - Config: {config.get_all_config()}")
    print(f"Před resetem - Log count: {logger.get_log_count()}")
    
    # Reset všech singletonů
    DatabaseConnection.reset()
    ConfigManager.reset()
    Logger.reset()
    
    # Vytvoření nových instancí
    new_db = DatabaseConnection()
    new_config = ConfigManager()
    new_logger = Logger()
    
    print(f"Po resetu - DB status: {new_db.get_status()}")
    print(f"Po resetu - Config: {new_config.get_all_config()}")
    print(f"Po resetu - Log count: {new_logger.get_log_count()}")


if __name__ == "__main__":
    demonstrate_singleton()
    demonstrate_multithread()
    demonstrate_reset_functionality()
