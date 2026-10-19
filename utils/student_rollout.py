"""Episode-level teacher/student mixture schedule (environment steps)."""


def student_probability(step, cfg):
    warmup = int(cfg['warmup_steps'])
    ramp = int(cfg['ramp_steps'])
    maximum = float(cfg.get('max_student_probability', 1.0))
    if warmup < 0 or ramp <= 0 or not 0 <= maximum <= 1:
        raise ValueError('Invalid student rollout schedule')
    return maximum * min(1.0, max(0.0, (step-warmup)/ramp))
