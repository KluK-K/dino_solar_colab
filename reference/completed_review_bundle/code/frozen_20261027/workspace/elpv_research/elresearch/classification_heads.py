"""Task heads shared by image-only and joint dense experiments."""
from torch import nn


HEAD_WIDTHS = {'linear': (), 'mlp128': (128,), 'mlp128_32': (128, 32)}


def head_spec(incoming, kind='linear'):
    if kind not in HEAD_WIDTHS:
        raise ValueError(f'Unknown classification head: {kind}')
    return {'kind': kind, 'input_dimensions': incoming,
            'hidden_dimensions': list(HEAD_WIDTHS[kind]), 'output_dimensions': 1,
            'hidden_activation': 'GELU' if HEAD_WIDTHS[kind] else None,
            'dropout': 0., 'output': 'binary logit; sigmoid only for prediction'}


def make_image_head(incoming, kind='linear'):
    widths = head_spec(incoming, kind)['hidden_dimensions']
    # Keep the legacy Linear state-dict keys for existing dense checkpoints.
    if not widths:
        return nn.Linear(incoming, 1)
    layers = []
    for outgoing in widths:
        layers.extend([nn.Linear(incoming, outgoing), nn.GELU()])
        incoming = outgoing
    return nn.Sequential(*layers, nn.Linear(incoming, 1))
