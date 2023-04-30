from os.path import dirname
from typing import Optional, Union
from drain3 import TemplateMiner  # type: ignore
from drain3.template_miner_config import TemplateMinerConfig  # type: ignore
from drain3.redis_persistence import RedisPersistence  # type: ignore
from drain3.file_persistence import FilePersistence  # type: ignore


class RedisParams:
    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: Optional[int] = None,
        redis_pass: Optional[str] = None,
        is_ssl: Optional[bool] = None,
        redis_key: Optional[str] = None,
    ) -> None:
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_pass = redis_pass
        self.is_ssl = is_ssl
        self.redis_key = redis_key


config = TemplateMinerConfig()
config.load(dirname(__file__) + "/../drain3.ini")
config.profiling_enabled = True


def create_template_miner(
    persistence_config: Union[RedisParams, str] = "drain3_state.json",
    drain3_config: TemplateMinerConfig = config,
) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.load(dirname(__file__) + "/../drain3.ini")
    config.profiling_enabled = True

    if isinstance(persistence_config, RedisParams):
        persistence = RedisPersistence(
            redis_host=persistence_config.redis_host,
            redis_port=persistence_config.redis_port,
            redis_db=persistence_config.redis_db,
            redis_pass=persistence_config.redis_pass,
            is_ssl=persistence_config.is_ssl,
            redis_key=persistence_config.redis_key,
        )
    else:
        persistence = FilePersistence(persistence_config)

    template_miner = TemplateMiner(persistence_handler=persistence, config=drain3_config)

    return template_miner
