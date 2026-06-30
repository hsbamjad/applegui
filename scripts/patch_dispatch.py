import re

with open('gui/main_window.py', 'r', encoding='utf-8') as f:
    txt = f.read()

old = (
    "            _MIN_DISPATCH_CONF = 0.28   # slightly below 0.30 to absorb float precision (0.2999... passes)\n"
    "            if self._sorter and self._sorting_enabled:\n"
    "                if rec.confidence >= _MIN_DISPATCH_CONF:\n"
    "                    self._sorter.schedule(\n"
    "                        apple_id   = rec.seq_id,\n"
    "                        lane       = rec.lane,\n"
    "                        grade      = rec.class_name,\n"
    "                        confidence = rec.confidence,\n"
    "                    )\n"
    "                else:\n"
    "                    log.warning(\n"
    '                        "Grade #%d rejected -- conf=%.2f < %.2f minimum",\n'
    "                        rec.seq_id, rec.confidence, _MIN_DISPATCH_CONF,\n"
    "                    )\n"
)

new = (
    "            # Dispatch gate: Fresh/Processing need minimum confidence to avoid\n"
    "            # false actuations. Cull is ALWAYS dispatched regardless of confidence\n"
    "            # because Cull = safe default - if sorter stays in last position (Fresh/\n"
    "            # Processing), the Cull apple ends up in the wrong bin.\n"
    "            _MIN_DISPATCH_CONF = 0.28\n"
    "            if self._sorter and self._sorting_enabled:\n"
    "                is_cull = rec.class_name == \"Cull\"\n"
    "                if is_cull or rec.confidence >= _MIN_DISPATCH_CONF:\n"
    "                    self._sorter.schedule(\n"
    "                        apple_id   = rec.seq_id,\n"
    "                        lane       = rec.lane,\n"
    "                        grade      = rec.class_name,\n"
    "                        confidence = rec.confidence,\n"
    "                    )\n"
    "                else:\n"
    "                    log.warning(\n"
    '                        "Grade #%d rejected -- conf=%.2f < %.2f minimum",\n'
    "                        rec.seq_id, rec.confidence, _MIN_DISPATCH_CONF,\n"
    "                    )\n"
)

if old in txt:
    txt = txt.replace(old, new, 1)
    with open('gui/main_window.py', 'w', encoding='utf-8') as f:
        f.write(txt)
    print('DONE')
else:
    print('NOT FOUND - checking current state:')
    idx = txt.find('_MIN_DISPATCH_CONF')
    print(repr(txt[idx-50:idx+300]))
