# Distance Landmark Selector

This script lets you interactively select **multiple two-point distance measurements** on an OpenEP/PyVista mesh. Each distance measurement consists of two picked mesh points.

The key behaviour is:

- You can save multiple two-point distances in one PyVista viewer session.
- Saved pairs remain visible while you continue selecting more distances.
- Landmarks are **not** added immediately when you save a pair.
- Once you close the PyVista window, all saved coordinates are added to the OpenEP case as landmarks.
- The selected coordinates are returned as a NumPy array.

---

## Requirements

You need the same Python environment used for your OpenEP/PyVista workflow.

Main dependencies:

```bash
pip install numpy pyvista vtk openep
```

---

## Basic Usage

Run the script directly:

```bash
python distance_landmark_selector_multi.py
```

The script will:

1. Create a PyVista mesh from the OpenEP case.
2. Add `bipolar_voltage` as a mesh scalar if it is available and has the correct size.
3. Open an interactive PyVista window.
4. Let you select and save multiple two-point distance pairs.
5. Add all saved coordinates as landmarks after the window is closed.
6. Return the selected coordinates as a NumPy array.

---

## Interactive Controls

| Action | Control |
|---|---|
| Select point 1 | Left click on the mesh |
| Select point 2 | Left click on the mesh again |
| View live distance before choosing point 2 | Move the mouse after selecting point 1 |
| Save the current two-point distance | Press `S` or click `SAVE PAIR` |
| Cancel the current unfinished pair | Press `R` or click `RESET / CANCEL` |
| Remove the most recently saved pair | Press `U` |
| Finish and add landmarks | Close the PyVista window |

---

## Recommended Workflow

1. Left click the first point on the mesh.
2. Move the mouse to preview the straight-line distance.
3. Left click the second point.
4. Press `S` or click `SAVE PAIR`.
5. Repeat steps 1–4 for every distance you want to measure.
6. Close the PyVista window.
7. The script will add all saved coordinates to the OpenEP case as landmarks.

A complete two-point pair that has not been saved manually is also included if you close the window after selecting both points.

For example, this is valid:

1. Click point 1.
2. Click point 2.
3. Close the window without pressing `S`.

That final pair will still be saved and added as landmarks.

---

## Landmark Naming

Each saved distance pair creates two landmarks.

The default naming pattern is:

```text
distance_landmark_D01_P1
distance_landmark_D01_P2
distance_landmark_D02_P1
distance_landmark_D02_P2
distance_landmark_D03_P1
distance_landmark_D03_P2
...
```

Where:

- `D01`, `D02`, `D03`, etc. are the saved distance pair numbers.
- `P1` and `P2` are the first and second point in that pair.

You can change the prefix by editing this argument in `main()`:

```python
selector = DistanceLandmarkSelector(
    case=case,
    mesh=pyvista_mesh,
    scalar_name=scalar_name,
    scalar_preference=scalar_preference,
    landmark_prefix="distance_landmark",
    display_unit="mm",
    close_on_select=False,
)
```

For example:

```python
landmark_prefix="mitral_distance"
```

would create names like:

```text
mitral_distance_D01_P1
mitral_distance_D01_P2
```

---

## Example Output

After closing the PyVista window, the script prints the selected coordinates and distances:

```text
Final selected coordinates, in mesh/case units:

Distance 1:
distance_landmark_D01_P1: [x1 y1 z1]
distance_landmark_D01_P2: [x2 y2 z2]
Distance: 12.34 mm

Distance 2:
distance_landmark_D02_P1: [x1 y1 z1]
distance_landmark_D02_P2: [x2 y2 z2]
Distance: 18.90 mm

Added 4 landmarks from 2 distance pair(s) using case.add_landmark(...).

Returned selected points array with shape:
(2, 2, 3)
```


## Important Notes

### Landmarks are added only after closing the viewer

Pressing `S` or `SAVE PAIR` stores the pair inside the selector, but does not immediately call:

```python
case.add_landmark(...)
```

The landmarks are added inside:

```python
add_saved_pairs_as_landmarks()
```

which is called after:

```python
self.plotter.show()
```

returns, meaning after the PyVista window has been closed.

---

### Reset does not delete saved pairs

`RESET / CANCEL` only clears the current unfinished selection.

It does not remove any pairs that you already saved with `S` or `SAVE PAIR`.

To remove the most recently saved pair, press:

```text
U
```

---

### Green markers are saved pairs

The script uses:

- Red markers/line for the current active pair
- Green markers/line for saved pairs

This helps distinguish the distance currently being selected from distances that have already been stored.

---