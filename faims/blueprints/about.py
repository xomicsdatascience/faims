from flask import Blueprint, current_app, render_template, session, redirect, url_for, request, flash, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
from faims.db import get_db, get_unique_value_for_field
import json
from os.path import join
import os
from faims.analysis import get_peptide_max_length, faims_cv_prediction, get_prediction_label
import numpy as np
import zipfile
import tempfile


bp = Blueprint("about", __name__, url_prefix="/about")

