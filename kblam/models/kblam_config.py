from transformers import PretrainedConfig


class KBLaMConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name_or_path: str = "",
        kb_layer_frequency: int = 3,
        kb_scale_factor: int | None = None, # Tỉ lệ điều chỉnh kích thước KB
        top_k_kb: int = 100,                # Số lượng top KB được sử dụng cho mỗi query trong attention
        dynamic_sparsify: bool = False,     # Nếu True, model sẽ lọc những KB quan trọng nhất cho mỗi query thay vì dùng toàn bộ từ KB
        sep_query_head: bool = False,       # Nếu True, model sẽ sử dụng một query head riêng biệt cho KB
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.kb_layer_frequency = kb_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.top_k_kb = top_k_kb
        self.dynamic_sparsify = dynamic_sparsify
        self.sep_query_head = sep_query_head
        self.attn_implementation = attn_implementation
        super().__init__(**kwargs)
