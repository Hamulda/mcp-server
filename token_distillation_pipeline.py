"""
Token Distillation Pipeline - nyní používá centrální TextProcessor
"""
from text_processing_utils import text_processor
import logging

class TokenDistillationPipeline:
    """Pipeline pro redukci tokenů v textu"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def distill_text(self, text: str, max_tokens: int = 3000) -> str:
        """Redukce textu pomocí centrálního procesoru"""
        return text_processor.optimize_for_tokens(text, max_tokens)

    def distill_medical_text(self, text: str, max_sentences: int = 10) -> str:
        """Destilace textu s prioritizací lékařských klíčových slov"""
        return text_processor.distill_medical_text(text, max_sentences)

    def log_pipeline_statistics(self, original_text: str, distilled_text: str):
        """Logování statistik pipeline"""
        original_tokens = text_processor.count_tokens(original_text)
        distilled_tokens = text_processor.count_tokens(distilled_text)
        reduction_percentage = ((original_tokens - distilled_tokens) / original_tokens) * 100 if original_tokens > 0 else 0

        self.logger.info(f"Původní počet tokenů: {original_tokens}")
        self.logger.info(f"Redukovaný počet tokenů: {distilled_tokens}")
        self.logger.info(f"Redukce: {reduction_percentage:.2f}%")
