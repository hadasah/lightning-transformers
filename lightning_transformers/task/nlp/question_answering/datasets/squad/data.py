# Copyright The PyTorch Lightning team.
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
from lightning_transformers.task.nlp.question_answering.data import QuestionAnsweringDataModule
from lightning_transformers.task.nlp.question_answering.datasets.squad.processing import (
    prepare_train_features,
    prepare_validation_features,
)


class SquadDataModule(QuestionAnsweringDataModule):

    @staticmethod
    def convert_to_train_features(*args, **kwargs):
        return prepare_train_features(*args, **kwargs)

    @staticmethod
    def convert_to_validation_features(*args, **kwargs):
        return prepare_validation_features(*args, **kwargs)
