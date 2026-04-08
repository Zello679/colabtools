# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom Jupyter notebook API handlers."""

import json
import os
import subprocess

from google.colab import _serverextension
from google.colab import drive
from google.colab._serverextension import _agent
from google.colab._serverextension import _resource_monitor
from jupyter_server.base import handlers
import tornado

_XSSI_PREFIX = ")]}'\n"


class ResourceUsageHandler(handlers.APIHandler):
  """Handles requests for memory usage of Colab kernels."""

  def initialize(self, kernel_manager):
    self._kernel_manager = kernel_manager

  @tornado.web.authenticated
  def get(self, *unused_args, **unused_kwargs):
    paths_str = os.getenv('COLAB_REPORT_USAGE_PATHS', '/')
    paths_list = paths_str.split(':')
    stats = _resource_monitor.get_resource_stats(
        self._kernel_manager, paths_list
    )
    self.set_header('Content-Type', 'application/json')
    self.finish(_XSSI_PREFIX + json.dumps(stats))


class DriveHandler(handlers.APIHandler):
  """Handles requests for drive errors."""

  def _get_drive_errors(self):
    """Reports errors from Drive.

    Returns:
      A list of strings describing evidence of unhealth, or [].
    """

    try:
      filtered_logfile = drive._timeouts_path()  # pylint:disable=protected-access
      if os.path.isfile(filtered_logfile):
        # Only return the most recent match since we only care to warn the user
        # about changes to this status.
        return [
            _serverextension._subprocess_check_output(  # pylint: disable=protected-access
                '/usr/bin/tail -1 "{}"'.format(filtered_logfile), shell=True
            )
            .decode('utf-8')
            .strip()
        ]
    except subprocess.CalledProcessError:  # Missing log file isn't fatal.
      pass

    return []

  @tornado.web.authenticated
  def get(self, *unused_args, **unused_kwargs):
    drive_status = self._get_drive_errors()
    self.finish(
        _XSSI_PREFIX
        + json.dumps({
            'dfs': drive_status,
        })
    )


class AgentHandler(handlers.APIHandler):
  """Handles requests for AI agent code generation."""

  def initialize(self, kernel_manager):
    self._kernel_manager = kernel_manager

  @tornado.web.authenticated
  async def post(self, *unused_args, **unused_kwargs):

    try:
      # Parse prompt from request JSON payload
      req_data = json.loads(self.request.body.decode('utf-8'))
      prompt = req_data.get('prompt', '')
      kernel_id = req_data.get('kernel_id')
      context = req_data.get('context', '')
      # Default to a generic session ID if not provided.
      session_id = req_data.get('session_id', 'session_1')
    except (json.JSONDecodeError, UnicodeDecodeError):
      prompt = ''
      context = ''
      kernel_id = None
      session_id = None

    try:
      agent_response = await _agent.send_message(
          prompt=prompt,
          context=context,
          kernel_manager=self._kernel_manager,
          kernel_id=kernel_id,
          session_id=session_id,
      )
    except (IOError, ValueError, TypeError, json.JSONDecodeError) as e:
      self.set_header('Content-Type', 'application/json')
      error_msg = f'{type(e).__name__} occurred during agent execution.'
      self.finish(
          _XSSI_PREFIX
          + json.dumps({'reply': 'An error occurred.', 'error': error_msg})
      )
      return

    self.set_header('Content-Type', 'application/json')
    self.finish(_XSSI_PREFIX + json.dumps(agent_response))


class AgentCreateHandler(handlers.APIHandler):
  """Handles requests for AI agent session creation."""

  def initialize(self, kernel_manager):
    self._kernel_manager = kernel_manager

  @tornado.web.authenticated
  async def post(self, *unused_args, **unused_kwargs):
    try:
      req_data = json.loads(self.request.body.decode('utf-8'))
      session_id = req_data.get('session_id', 'session_1')
      instructions = req_data.get('instructions')
    except (json.JSONDecodeError, UnicodeDecodeError):
      session_id = 'session_1'
      instructions = None

    _agent.create_session(session_id, instructions)
    self.set_header('Content-Type', 'application/json')
    self.finish(_XSSI_PREFIX + json.dumps({'status': 'ok'}))


class BuildInfoHandler(handlers.APIHandler):
  """Handles requests for build info of the Colab kernel."""

  @tornado.web.authenticated
  def get(self, *unused_args, **unused_kwargs):
    self.set_header('Content-Type', 'application/json')
    self.finish(
        _XSSI_PREFIX
        + json.dumps({
            'release_tag': os.environ.get('COLAB_RELEASE_TAG'),
        })
    )
