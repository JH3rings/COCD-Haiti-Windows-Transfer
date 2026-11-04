"""Isolated training recipe; does not modify historical model constructors."""
import math
import torch
from timm.layers import DropPath
from windows_main.models_pico_target_search import PicoStudent, PicoTeacher


def configure_stochastic_depth(model):
    blocks = [block for stage in model.backbone.stages for block in stage.blocks]
    for i, block in enumerate(blocks):
        rate = .05 * i / max(len(blocks) - 1, 1)
        block.drop_path = DropPath(rate) if rate else torch.nn.Identity()
    return model


class Student(PicoStudent):
    def __init__(self):
        super().__init__()
        configure_stochastic_depth(self)


class Teacher(PicoTeacher):
    def __init__(self):
        super().__init__()
        configure_stochastic_depth(self)


def make_optimizer(model):
    exempt = set()
    for module in model.modules():
        kind = type(module).__name__.lower()
        if any(tag in kind for tag in ('norm', 'grn', 'globalresponse')):
            exempt.update(id(p) for p in module.parameters(recurse=False))
    groups = {}
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert id(param) not in seen
        seen.add(id(param))
        encoder = name.startswith(('backbone.stem.', 'backbone.stages.'))
        no_decay = param.ndim <= 1 or name.endswith('.bias') or id(param) in exempt
        key = ('encoder' if encoder else 'new') + ('_no_decay' if no_decay else '_decay')
        if key not in groups:
            lr = 2e-5 if encoder else 1e-4
            groups[key] = dict(params=[], names=[], group_name=key, lr=lr, peak_lr=lr,
                               weight_decay=0. if no_decay else .01, schedule_step=0)
        groups[key]['params'].append(param)
        groups[key]['names'].append(name)
    assert seen == {id(p) for p in model.parameters() if p.requires_grad}
    return torch.optim.AdamW(list(groups.values()), betas=(.9, .999), eps=1e-8)


def lr_factor(step, steps_per_epoch):
    h, t = 50 * steps_per_epoch, 100 * steps_per_epoch
    if not 0 <= step < t:
        raise ValueError(f'Optimizer step {step} outside the declared 100 epochs')
    return 1. if step < h else .01 + .99 * (1 + math.cos(math.pi * (step-h)/(t-1-h))) / 2


def schedule_before_step(optimizer, steps_per_epoch=225):
    s = optimizer.param_groups[0]['schedule_step']
    for group in optimizer.param_groups:
        assert group['schedule_step'] == s
        group['lr'] = group['peak_lr'] * lr_factor(s, steps_per_epoch)
        group['schedule_step'] = s + 1


def optimizer_record(optimizer):
    return [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]
