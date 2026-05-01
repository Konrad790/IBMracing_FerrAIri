# TORCS Metrics Audit

Zakres: obecna metoda strojenia `python autoresearch.py --strategy documented-turns`.

`Bierzemy pod uwage` = aktualny driver albo evaluator odczytuje to pole lub wyprowadza z niego metryke pochodna.

`Optymalizujemy` = to pole albo metryka pochodna z niego wyprowadzona wplywa bezposrednio na ranking kandydata albo na twarde odrzucenie runu.

| Metryka TORCS | Bierzemy pod uwage | Optymalizujemy |
| --- | --- | --- |
| `angle` | tak | tak |
| `curLapTime` | tak | tak |
| `damage` | tak | tak |
| `distFromStart` | tak | nie |
| `distRaced` | tak | tak |
| `focus` | nie | nie |
| `fuel` | nie | nie |
| `gear` | tak | nie |
| `lastLapTime` | tak | tak |
| `opponents` | nie | nie |
| `racePos` | nie | nie |
| `rpm` | tak | nie |
| `speedX` | tak | tak |
| `speedY` | nie | nie |
| `speedZ` | nie | nie |
| `track` | tak | tak |
| `trackPos` | tak | tak |
| `wheelSpinVel` | tak | nie |
| `z` | nie | nie |

Uwaga: glowna funkcja celu nie dziala bezposrednio na wszystkich surowych sensorach TORCS. W praktyce ranking kandydatow opiera sie glownie na metrykach pochodnych takich jak `lap_time`, `current_lap_time`, `dist_raced`, `damage`, `offtrack_ticks`, `max_abs_track_pos`, `slow_ticks`, `backwards_ticks` i `sector_times`.
