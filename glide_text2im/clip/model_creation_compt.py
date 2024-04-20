from glide_text2im.clip.model_creation import *


@attr.s
class CLIPComptModel:
    config: Dict[str, Any] = attr.ib()
    text_encoder: nn.Module = attr.ib()
    image_encoder: nn.Module = attr.ib()
    logit_scale: torch.Tensor = attr.ib()
    device: torch.device = attr.ib()
    tokenizer: SimpleTokenizer = attr.ib()

    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lens = []
        for prompt in prompts:
            sub_tokens, sub_len = self.tokenizer.padded_tokens_and_len(
                self.tokenizer.encode(prompt), self.text_encoder.max_text_len
            )
            tokens.append(sub_tokens)
            lens.append(sub_len)
        return (
            torch.tensor(tokens).to(dtype=torch.long, device=self.device),
            torch.tensor(lens).to(dtype=torch.long, device=self.device),
        )

    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens, lens = self.encode_prompts(prompts)
        z_t = self.text_encoder(tokens, lens)
        return z_t / (torch.linalg.norm(z_t, dim=-1, keepdim=True) + 1e-12)

    def image_embeddings(self, images: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_i = self.image_encoder((images + 1) * 127.5, t)
        return z_i / (torch.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-12)

    def cond_fn(self, prompts: List[str], grad_scale: float, guidance_steps, respace_gap) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)

        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            resp_timestep = int((t[0] + 1) / respace_gap)

            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                if guidance_steps[resp_timestep - 1] == 1:
                    print(resp_timestep)
                    z_i = self.image_embeddings(x_var, t)
                    loss = torch.exp(self.logit_scale) * (z_t * z_i).sum()
                    grad = torch.autograd.grad(loss, x_var)[0].detach()
                else:
                    grad = torch.zeros_like(x_var)
            return grad * grad_scale
        return cond_fn

def create_clip_compt_model(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> CLIPComptModel:
    if config_path is None:
        config_path = default_config_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    text_encoder = TextEncoder(
        n_bpe_vocab=config["n_vocab"],
        max_text_len=config["max_text_len"],
        n_embd=config["n_embd"],
        n_head=config["n_head_text"],
        n_xf_blocks=config["n_xf_blocks_text"],
        n_head_state=config["n_head_state_text"],
        device=device,
    )

    image_encoder = ImageEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head_image"],
        n_xf_blocks=config["n_xf_blocks_image"],
        n_head_state=config["n_head_state_image"],
        n_timestep=config["n_timesteps"],
        device=device,
    )

    logit_scale = torch.tensor(
        np.log(config["logit_scale"]),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return CLIPComptModel(
        config=config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        logit_scale=logit_scale,
        device=device,
        tokenizer=tokenizer,
    )
