import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        # 설정 파일 불러오고, 불러올 체크포인트 수정
        self._config = _update_config(config, modification)
        self.resume = resume

        # 저장할 디렉터리 설정 (이름 / model, log)
        save_dir = Path(self.config['trainer']['save_dir']) # 저장할 경로 설정

        exper_name = self.config['name']
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # 저장할 디렉터리 확인 및 생성
        exist_ok = run_id == '' # 이미 존재하는 디렉터리 에러 발생 X
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 업데이트된 config (설정 파일) 체크포인트 디렉터리에 저장
        write_json(self.config, self.save_dir / 'config.json')

        # 로그 설정 초기화, 레벨 설정
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        # 옵션 추가
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        
        # args parsing
        if not isinstance(args, tuple): # 'args'가 tuple이 아닌 경우 method 호출을 통해 parsing
            args = args.parse_args()
        
        # config 파일 로드
        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        cfg_fname = Path(args.config)

        # 설정 파일 읽기 및 업데이트
        config = read_json(cfg_fname)
        if args.config:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # 수정사항 파싱
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        # 인스턴스 생성 및 반환
        return cls(config, modification)

    # 객체 및 함수 초기화
    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type'] # name 키의 type
        module_args = dict(self[name]['args']) # name 키의 args
        # kwargs에 있는 키가 module_args에 없는지 확인, 있으면 예외 발생
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs) # module_args -> kwargs 업데이트
        return getattr(module, module_name)(*args, **module_args) # module_name 함수 가져와 인수 초기화
    

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args) # partial 객체 생성, 인수 고정한 함수 반환

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        # 유효성 검사, log_level 기반
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name) # 로거 가져오기
        logger.setLevel(self.log_levels[verbosity]) # 로깅 레벨 설정
        return logger # 설정된 로거 반환

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
# config -> modification 업데이트
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config
# 옵션 이름에서 '--' 제거
def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')
# 키 경로를 따라 중첩된 객체 처리
def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value
    
def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
