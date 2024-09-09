__version__ = "0.2.0"

from flask import Flask
import os
from os.path import join, dirname, basename
import shutil
import json


def create_app(test_config=None) -> Flask:
    """Creates the app and returns it."""
    env_dict = parse_env()

    instance_path = env_dict["INSTANCE_PATH"]
    static_directory = join(instance_path, "static")
    template_directory = join(instance_path, "templates")

    this_dir = dirname(dirname(__file__))
    template_source = join(this_dir, "templates")
    static_source = join(this_dir, "static")
    print(f"Copying from {template_source} into {template_directory}")
    shutil.copytree(template_source, template_directory, dirs_exist_ok=True)
    print(f"Copying from {static_source} into {static_directory}")
    shutil.copytree(static_source, static_directory, dirs_exist_ok=True)

    os.makedirs(join(instance_path, env_dict["UPLOAD_DIRECTORY"]), exist_ok=True)
    os.makedirs(join(instance_path, env_dict["RESULTS_DIRECTORY"]), exist_ok=True)

    app = Flask(__name__,
                instance_path=env_dict["INSTANCE_PATH"],
                instance_relative_config=True,
                template_folder=template_directory,
                static_folder=static_directory)

    app.config.from_mapping(DATABASE=join(app.instance_path, "faims.sqlite"),
                            **env_dict)
    app.config["UPLOAD_DIRECTORY"] = join(instance_path, env_dict["UPLOAD_DIRECTORY"])
    app.config["RESULTS_DIRECTORY"] = join(instance_path, env_dict["RESULTS_DIRECTORY"])
    app.config["MODEL_FILE"] = join(instance_path, env_dict["MODEL_FILE"])
    from . import db
    db.init_app(app)

    app.jinja_env.filters['basename'] = basename
    app.jinja_env.filters['dirname'] = dirname

    from faims.blueprints import landing
    app.register_blueprint(landing.bp)

    app.add_url_rule('/', endpoint='index')
    return app


def parse_env(env_file: str = '.env') -> dict:
    """
    Parses the environment file and returns the result as a dict.
    Parameters
    ----------
    env_file : str
        Path to the environment

    Returns
    -------
    dict
        Dictionary keyed with the environment variable name.
    """
    f = open(env_file)
    env_dict = dict()
    line = f.readline()
    line_count = 0  # on the off chance that the problematic line contains the secret key, don't print contents
    while len(line) > 0:
        line_count += 1
        idx_equal = line.index('=')
        if idx_equal == -1:
            raise ValueError(f"Environment file not correctly configured; see line {line_count} in .env file.")
        key = line[:idx_equal]
        env_dict[key] = json.loads(line[idx_equal+1:])
        line = f.readline()
    f.close()
    return env_dict
