import numpy as np
import pyvista as pv
import openep

from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkLineSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkTextActor,
)


try:
    case = cases[case_1]
except NameError:
    OPENEP_FILE = "/Users/vinush-vigneswaran/Documents/09_DATASETS/openep_dataset_2.mat"
    out_cases = {}
    case = openep.load_openep_mat(
        filename=f"{OPENEP_FILE}",
    )
    case_1 = case.name
    colorbar_upper = 2.0
    colorbar_lower = 0.0


def add_bipolar_voltage_to_mesh(case, mesh, field_name="bipolar_voltage"):
    """
    Adds case.fields.bipolar_voltage to the PyVista mesh if it exists and has
    a usable size.

    Handles the common missing-field case where bipolar_voltage has size 1.
    Returns:
        (scalar_name, preference)
        scalar_name is None if no usable field was added.
        preference is 'point' or 'cell' when added.
    """
    fields = getattr(case, "fields", None)
    voltage = getattr(fields, field_name, None)

    if voltage is None:
        print("No bipolar_voltage field found on case.fields.")
        return None, None

    voltage = np.asarray(voltage).squeeze()

    if voltage.size <= 1:
        print(
            "bipolar_voltage exists but appears to be empty/missing "
            f"(size={voltage.size}). Mesh will be shown without this scalar."
        )
        return None, None

    voltage = voltage.ravel().astype(float)

    if voltage.size == mesh.n_points:
        mesh.point_data[field_name] = voltage
        print(f"Added {field_name} as point_data with {voltage.size} values.")
        return field_name, "point"

    if voltage.size == mesh.n_cells:
        mesh.cell_data[field_name] = voltage
        print(f"Added {field_name} as cell_data with {voltage.size} values.")
        return field_name, "cell"

    print(
        f"bipolar_voltage size mismatch: got {voltage.size}, "
        f"but mesh has {mesh.n_points} points and {mesh.n_cells} cells. "
        "Mesh will be shown without this scalar."
    )
    return None, None


class DistanceLandmarkSelector:
    def __init__(
        self,
        case,
        mesh,
        scalar_name=None,
        scalar_preference=None,
        landmark_prefix="distance_landmark",
        display_scale=1000.0,
        display_unit="mm",
        close_on_select=False,
    ):
        """
        Select multiple two-point distance measurements on a PyVista mesh.

        Workflow:
            1. Left click point 1.
            2. Left click point 2.
            3. Press SAVE PAIR/S to store that two-point distance.
            4. Repeat for as many distances as needed.
            5. Close the PyVista window.

        Landmarks are only added to the OpenEP case after the PyVista window is
        closed. Each saved pair is added as two landmarks:
            {landmark_prefix}_D01_P1, {landmark_prefix}_D01_P2,
            {landmark_prefix}_D02_P1, {landmark_prefix}_D02_P2, etc.

        Args:
            case:
                OpenEP case object.
            mesh:
                PyVista mesh from case.create_mesh().
            scalar_name:
                Optional scalar to colour the mesh by.
            scalar_preference:
                'point' or 'cell'.
            landmark_prefix:
                Prefix used for the OpenEP landmarks.
            display_scale:
                Distance display multiplier. Since your load uses scale_points=1e-3,
                mesh coordinates are likely metres, so 1000 displays mm.
            display_unit:
                Unit text shown beside live distance.
            close_on_select:
                If True, the PyVista window closes after pressing SAVE PAIR/S.
                Keep False when selecting multiple distances.
        """
        self.case = case
        self.mesh = mesh
        self.scalar_name = scalar_name
        self.scalar_preference = scalar_preference
        self.landmark_prefix = landmark_prefix
        self.display_scale = display_scale
        self.display_unit = display_unit
        self.close_on_select = close_on_select

        self.first_point = None
        self.second_point = None
        self.hover_point = None

        # Saved two-point distances. Each entry is a dict with keys:
        # index, point_1, point_2, raw_distance, display_distance.
        self.selected_pairs = []
        self.completed_measurement_actors = []
        self.landmarks_added = False

        self.plotter = pv.Plotter(notebook=False)
        self.renderer = self.plotter.renderer

        self.marker_radius = self._estimate_marker_radius()

        self._add_mesh()
        self._setup_picker()
        self._setup_dynamic_actors()
        self._setup_text_and_buttons()
        self._setup_events()

    def _estimate_marker_radius(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        diagonal = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])

        if diagonal <= 0:
            return 1e-3

        return diagonal * 0.008

    def _add_mesh(self):
        mesh_kwargs = dict(
            pickable=True,
            show_edges=False,
        )

        if self.scalar_name is not None:
            mesh_kwargs.update(
                scalars=self.scalar_name,
                preference=self.scalar_preference,
                cmap="jet_r",
                nan_color="grey",
                scalar_bar_args={"title": self.scalar_name},
                clim=(colorbar_lower, colorbar_upper),
            )
        else:
            mesh_kwargs.update(color="lightgrey")

        self.mesh_actor = self.plotter.add_mesh(self.mesh, **mesh_kwargs)

    def _setup_picker(self):
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)

        # Restrict picking to the anatomical mesh, not the marker balls/lines.
        self.picker.AddPickList(self.mesh_actor)
        self.picker.PickFromListOn()

    def _make_sphere_actor(self, opacity=1.0, color=(1.0, 0.0, 0.0)):
        source = vtkSphereSource()
        source.SetCenter(0.0, 0.0, 0.0)
        source.SetRadius(self.marker_radius)
        source.SetThetaResolution(24)
        source.SetPhiResolution(24)
        source.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.SetVisibility(False)

        return actor

    def _make_line_actor(self, point_1, point_2, color=(0.0, 0.8, 0.0), line_width=3):
        line_source = vtkLineSource()
        line_source.SetPoint1(float(point_1[0]), float(point_1[1]), float(point_1[2]))
        line_source.SetPoint2(float(point_2[0]), float(point_2[1]), float(point_2[2]))
        line_source.Update()

        line_mapper = vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_source.GetOutputPort())

        line_actor = vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(*color)
        line_actor.GetProperty().SetLineWidth(line_width)
        line_actor.SetVisibility(True)

        return line_actor, line_source

    def _setup_dynamic_actors(self):
        # Red actors show the currently active, not-yet-saved pair.
        self.first_marker_actor = self._make_sphere_actor(opacity=1.0, color=(1.0, 0.0, 0.0))
        self.second_marker_actor = self._make_sphere_actor(opacity=1.0, color=(1.0, 0.0, 0.0))
        self.hover_marker_actor = self._make_sphere_actor(opacity=0.55, color=(1.0, 0.0, 0.0))

        self.renderer.AddActor(self.first_marker_actor)
        self.renderer.AddActor(self.second_marker_actor)
        self.renderer.AddActor(self.hover_marker_actor)

        self.line_source = vtkLineSource()
        self.line_mapper = vtkPolyDataMapper()
        self.line_mapper.SetInputConnection(self.line_source.GetOutputPort())

        self.line_actor = vtkActor()
        self.line_actor.SetMapper(self.line_mapper)
        self.line_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # red
        self.line_actor.GetProperty().SetLineWidth(4)
        self.line_actor.SetVisibility(False)

        self.renderer.AddActor(self.line_actor)

        self.distance_text_actor = vtkTextActor()
        self.distance_text_actor.GetTextProperty().SetFontSize(18)
        self.distance_text_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
        self.distance_text_actor.GetTextProperty().SetBold(True)
        self.distance_text_actor.SetVisibility(False)

        self.renderer.AddActor2D(self.distance_text_actor)

    def _setup_text_and_buttons(self):
        self.plotter.add_text(
            (
                "Left click: choose point 1, then point 2\n"
                "Move mouse after point 1: live straight-line distance\n"
                "SAVE PAIR/S: store current two-point distance and continue\n"
                "UNDO/U: remove last saved pair\n"
                "RESET/R: cancel current unfinished pair\n"
                "Close window: add all saved coordinates as landmarks"
            ),
            position=(10, 150),
            font_size=10,
            name="help_text",
        )

        self._set_status("Click the first point on the mesh.")

        self.select_button = self.plotter.add_checkbox_button_widget(
            self._on_select_button,
            value=False,
            position=(10, 70),
            size=30,
            color_on="green",
            color_off="white",
            background_color="white",
        )
        self.plotter.add_text(
            "SAVE PAIR",
            position=(50, 75),
            font_size=10,
            name="select_label",
        )

        self.reset_button = self.plotter.add_checkbox_button_widget(
            self._on_reset_button,
            value=False,
            position=(10, 30),
            size=30,
            color_on="red",
            color_off="white",
            background_color="white",
        )
        self.plotter.add_text(
            "RESET / CANCEL",
            position=(50, 35),
            font_size=10,
            name="reset_label",
        )

    def _setup_events(self):
        self.plotter.iren.add_observer("MouseMoveEvent", self._on_mouse_move)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_click)

        self.plotter.add_key_event("s", self.save_current_pair)
        self.plotter.add_key_event("u", self.undo_last_pair)
        self.plotter.add_key_event("r", self.reset_selection)

    def _set_status(self, message):
        self.plotter.remove_actor("status_text", render=False)
        self.plotter.add_text(
            message,
            position=(10, 125),
            font_size=10,
            name="status_text",
        )

    def _untick_button(self, button):
        try:
            button.GetRepresentation().SetState(0)
        except Exception:
            pass

    def _on_select_button(self, state):
        if state:
            self.save_current_pair()
            self._untick_button(self.select_button)

    def _on_reset_button(self, state):
        if state:
            self.reset_selection()
            self._untick_button(self.reset_button)

    def _mouse_is_over_ui(self, x, y):
        """
        Prevent clicking SAVE/RESET from also selecting a mesh point behind
        the UI widgets. Coordinates are approximate screen-pixel boxes.
        """
        ui_boxes = [
            (0, 20, 210, 50),   # RESET area
            (0, 60, 210, 50),   # SAVE PAIR area
        ]

        for x0, y0, width, height in ui_boxes:
            if x0 <= x <= x0 + width and y0 <= y <= y0 + height:
                return True

        return False

    def _pick_surface_at_mouse(self):
        interactor = self.plotter.iren.interactor
        x, y = interactor.GetEventPosition()

        if self._mouse_is_over_ui(x, y):
            return None, x, y

        picked = self.picker.Pick(x, y, 0, self.renderer)

        if not picked:
            return None, x, y

        point = np.array(self.picker.GetPickPosition(), dtype=float)
        return point, x, y

    def _set_marker(self, actor, point, visible=True):
        actor.SetPosition(float(point[0]), float(point[1]), float(point[2]))
        actor.SetVisibility(bool(visible))

    def _set_line(self, point_1, point_2, visible=True):
        self.line_source.SetPoint1(
            float(point_1[0]),
            float(point_1[1]),
            float(point_1[2]),
        )
        self.line_source.SetPoint2(
            float(point_2[0]),
            float(point_2[1]),
            float(point_2[2]),
        )
        self.line_source.Modified()
        self.line_actor.SetVisibility(bool(visible))

    def _distance(self, point_1, point_2):
        return float(np.linalg.norm(np.asarray(point_2) - np.asarray(point_1)))

    def _on_mouse_move(self, *args):
        if self.first_point is None:
            return

        if self.second_point is not None:
            return

        picked_point, x, y = self._pick_surface_at_mouse()

        if picked_point is None:
            self.hover_marker_actor.SetVisibility(False)
            self.distance_text_actor.SetVisibility(False)
            self.plotter.render()
            return

        self.hover_point = picked_point

        self._set_marker(self.hover_marker_actor, picked_point, visible=True)
        self._set_line(self.first_point, picked_point, visible=True)

        raw_distance = self._distance(self.first_point, picked_point)
        display_distance = raw_distance * self.display_scale

        self.distance_text_actor.SetInput(
            f"{display_distance:.2f} {self.display_unit}"
        )
        self.distance_text_actor.SetPosition(int(x) + 18, int(y) + 18)
        self.distance_text_actor.SetVisibility(True)

        self.plotter.render()

    def _on_left_click(self, *args):
        picked_point, x, y = self._pick_surface_at_mouse()

        if picked_point is None:
            return

        if self.first_point is None:
            self.first_point = picked_point
            self._set_marker(self.first_marker_actor, self.first_point, visible=True)
            self._set_status("First point selected. Move mouse to see distance, then click second point.")
            self.plotter.render()
            return

        if self.second_point is None:
            self.second_point = picked_point
            self._set_marker(self.second_marker_actor, self.second_point, visible=True)
            self._set_line(self.first_point, self.second_point, visible=True)

            self.hover_marker_actor.SetVisibility(False)
            self.distance_text_actor.SetVisibility(False)

            raw_distance = self._distance(self.first_point, self.second_point)
            display_distance = raw_distance * self.display_scale

            self._set_status(
                f"Second point selected. Distance = {display_distance:.2f} {self.display_unit}. "
                "Press SAVE PAIR/S to store this distance, or close the window to store it automatically."
            )
            self.plotter.render()
            return

        self._set_status(
            "Two points are already selected. Press SAVE PAIR/S to store them, "
            "or RESET/R to choose this pair again."
        )

    def _clear_current_dynamic_selection(self):
        self.first_point = None
        self.second_point = None
        self.hover_point = None

        self.first_marker_actor.SetVisibility(False)
        self.second_marker_actor.SetVisibility(False)
        self.hover_marker_actor.SetVisibility(False)
        self.line_actor.SetVisibility(False)
        self.distance_text_actor.SetVisibility(False)

    def _add_fixed_pair_actors(self, point_1, point_2):
        """Keep a saved pair visible while the user selects more distances."""
        fixed_marker_colour = (0.0, 0.8, 0.0)
        fixed_line_colour = (0.0, 0.8, 0.0)

        marker_1 = self._make_sphere_actor(opacity=1.0, color=fixed_marker_colour)
        marker_2 = self._make_sphere_actor(opacity=1.0, color=fixed_marker_colour)
        self._set_marker(marker_1, point_1, visible=True)
        self._set_marker(marker_2, point_2, visible=True)

        line_actor, line_source = self._make_line_actor(
            point_1,
            point_2,
            color=fixed_line_colour,
            line_width=3,
        )

        self.renderer.AddActor(marker_1)
        self.renderer.AddActor(marker_2)
        self.renderer.AddActor(line_actor)

        actors = {
            "marker_1": marker_1,
            "marker_2": marker_2,
            "line_actor": line_actor,
            "line_source": line_source,
        }
        self.completed_measurement_actors.append(actors)

    def save_current_pair(self):
        """
        Store the currently selected two-point distance, but do not add
        landmarks to the case yet. Landmarks are added after the window closes.
        """
        if self.first_point is None or self.second_point is None:
            self._set_status("Select two mesh points before pressing SAVE PAIR.")
            self.plotter.render()
            return None

        point_1 = np.asarray(self.first_point, dtype=float).copy()
        point_2 = np.asarray(self.second_point, dtype=float).copy()
        raw_distance = self._distance(point_1, point_2)
        display_distance = raw_distance * self.display_scale

        pair_index = len(self.selected_pairs) + 1
        pair = {
            "index": pair_index,
            "point_1": point_1,
            "point_2": point_2,
            "raw_distance": raw_distance,
            "display_distance": display_distance,
        }
        self.selected_pairs.append(pair)

        self._add_fixed_pair_actors(point_1, point_2)
        self._clear_current_dynamic_selection()

        self._set_status(
            f"Saved distance {pair_index}: {display_distance:.2f} {self.display_unit}. "
            "Click the first point for the next distance, or close the window when finished."
        )
        self.plotter.render()

        if self.close_on_select:
            self.plotter.close()

        return pair

    # Backwards-compatible name for your old SELECT/S workflow.
    def finalise_selection(self):
        return self.save_current_pair()

    def reset_selection(self):
        """Cancel only the current unfinished pair; saved pairs are kept."""
        self._clear_current_dynamic_selection()
        self._set_status(
            f"Current selection reset. {len(self.selected_pairs)} saved pair(s) kept. "
            "Click the first point on the mesh."
        )
        self.plotter.render()

    def undo_last_pair(self):
        """Remove the most recently saved pair and its permanent actors."""
        if not self.selected_pairs:
            self._set_status("There are no saved pairs to undo.")
            self.plotter.render()
            return None

        removed_pair = self.selected_pairs.pop()

        actors = self.completed_measurement_actors.pop()
        for key in ("marker_1", "marker_2", "line_actor"):
            self.renderer.RemoveActor(actors[key])

        self._set_status(
            f"Removed saved distance {removed_pair['index']}. "
            f"{len(self.selected_pairs)} saved pair(s) remain."
        )
        self.plotter.render()
        return removed_pair

    def _save_current_pair_if_complete(self):
        """
        If the user selected two points and then closed the window without
        pressing SAVE PAIR/S, still include that final pair.
        """
        if self.first_point is not None and self.second_point is not None:
            point_1 = np.asarray(self.first_point, dtype=float).copy()
            point_2 = np.asarray(self.second_point, dtype=float).copy()
            raw_distance = self._distance(point_1, point_2)
            display_distance = raw_distance * self.display_scale

            pair_index = len(self.selected_pairs) + 1
            self.selected_pairs.append(
                {
                    "index": pair_index,
                    "point_1": point_1,
                    "point_2": point_2,
                    "raw_distance": raw_distance,
                    "display_distance": display_distance,
                }
            )
            self._clear_current_dynamic_selection()

    def add_saved_pairs_as_landmarks(self):
        """Add all saved two-point distances to the OpenEP case as landmarks."""
        if self.landmarks_added:
            print("Landmarks have already been added for this selector run.")
            return self.get_selected_points_array()

        if not self.selected_pairs:
            print("No saved distance pairs. No landmarks were added.")
            return None

        print("\nFinal selected coordinates, in mesh/case units:")

        for pair in self.selected_pairs:
            pair_index = pair["index"]
            name_1 = f"{self.landmark_prefix}_D{pair_index:02d}_P1"
            name_2 = f"{self.landmark_prefix}_D{pair_index:02d}_P2"

            self.case.add_landmark(
                name=name_1,
                internal_name=name_1,
                point=np.asarray(pair["point_1"], dtype=float),
            )
            self.case.add_landmark(
                name=name_2,
                internal_name=name_2,
                point=np.asarray(pair["point_2"], dtype=float),
            )

            print(f"\nDistance {pair_index}:")
            print(f"{name_1}: {pair['point_1']}")
            print(f"{name_2}: {pair['point_2']}")
            print(f"Distance: {pair['display_distance']:.2f} {self.display_unit}")

        self.landmarks_added = True
        print(
            f"\nAdded {len(self.selected_pairs) * 2} landmarks "
            f"from {len(self.selected_pairs)} distance pair(s) using case.add_landmark(...)."
        )

        return self.get_selected_points_array()

    def get_selected_points_array(self):
        """
        Return selected coordinates as a NumPy array with shape:
            (number_of_distances, 2, 3)

        Axis 1 is [point_1, point_2]. Axis 2 is [x, y, z].
        """
        if not self.selected_pairs:
            return None

        return np.asarray(
            [
                [pair["point_1"], pair["point_2"]]
                for pair in self.selected_pairs
            ],
            dtype=float,
        )

    def get_measurement_summary(self):
        """Return distances and coordinates as plain Python dictionaries."""
        return [
            {
                "index": pair["index"],
                "point_1": pair["point_1"].tolist(),
                "point_2": pair["point_2"].tolist(),
                "raw_distance": pair["raw_distance"],
                "display_distance": pair["display_distance"],
                "display_unit": self.display_unit,
            }
            for pair in self.selected_pairs
        ]

    def run(self):
        self.plotter.show()

        # When the user closes the window, include any complete but unsaved pair,
        # then add every saved pair as landmarks.
        self._save_current_pair_if_complete()
        selected_points = self.add_saved_pairs_as_landmarks()
        out_cases[f'{case_1}_distance_landmarks'] = self.case

        return selected_points


def main():
    pyvista_mesh = case.create_mesh()

    scalar_name, scalar_preference = add_bipolar_voltage_to_mesh(
        case,
        pyvista_mesh,
        field_name="bipolar_voltage",
    )

    selector = DistanceLandmarkSelector(
        case=case,
        mesh=pyvista_mesh,
        scalar_name=scalar_name,
        scalar_preference=scalar_preference,
        landmark_prefix="distance_landmark",
        display_unit="mm",
        close_on_select=False,
    )

    selected_points = selector.run()

    if selected_points is None:
        print("No distance pairs were selected.")
    else:
        print("\nReturned selected points array with shape:")
        print(selected_points.shape)  # (number_of_distances, 2, 3)
        print("\nReturned selected points:")
        print(selected_points)

    return selected_points


if __name__ == "__main__":
    selected_points = main()
