# Copyright Larry. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.jingyu import Jingyu


@register_model("jingyu2_1")
class Jingyu2_1(Jingyu):
    """Jingyu2.1 model wrapper.

    Uses the same implementation as Jingyu since both share identical interfaces
    via standard Transformers Auto* loading and checkpoint ``auto_map``.
    """

    def __init__(self, pretrained: str, **kwargs):
        """Initialize the Jingyu2.1 wrapper by delegating to ``Jingyu``.

        Args:
            pretrained (str): Hugging Face model id or local checkpoint path.
            **kwargs: Additional keyword arguments forwarded to ``Jingyu.__init__``.
        """
        super().__init__(pretrained=pretrained, **kwargs)
