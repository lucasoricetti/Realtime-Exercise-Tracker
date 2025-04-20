"""
AIGym_Modified - A real-time pose-based fitness tracker for recognizing squats and push-ups.

Uses YOLO pose keypoints to classify exercises and count repetitions based on body angles.
Supports multiple people simultaneously with automatic exercise detection.
"""

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults

class AIGym_Modified(BaseSolution):
    """
    Extended solution class for recognizing and counting squats and push-ups in a video stream.

    This class builds on the BaseSolution to detect poses using the YOLO pose model and classify exercises
    as either squats or push-ups. It tracks each individual's progress through exercise stages
    ('up' and 'down') and counts repetitions per person, supporting multiple people doing different exercises.

    Attributes:
        squat_count (List[int]): Squat repetition counts for each tracked person.
        pushup_count (List[int]): Push-up repetition counts for each tracked person.
        angle (List[float]): Current joint angle used to determine exercise stage.
        stage (List[str]): Current stage of exercise for each person ('up', 'down', or '-').
        exercise (List[str | None]): Currently detected exercise type ('squat' or 'pushup') per person.
        up_angle (float): Angle threshold representing the "up" position.
        down_angle (float): Angle threshold representing the "down" position.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AIGym_Modified class with custom thresholds and data tracking lists.

        Args:
            **kwargs (Any): Configuration options passed to BaseSolution.
        """
        super().__init__(**kwargs)
        self.squat_count = []
        self.pushup_count = []
        self.angle = []
        self.stage = []
        self.exercise = []

        # Load angle thresholds from the configuration
        self.up_angle = float(self.CFG.get("up_angle", 160))
        self.down_angle = float(self.CFG.get("down_angle", 90))

    def detect_exercise(self, kpts):
        """
        Automatically classify the exercise as 'squat' or 'pushup' based on keypoint geometry.

        Args:
            kpts (List[List[float]]): Keypoints for a person (x, y, confidence).

        Returns:
            str | None: Returns 'squat', 'pushup', or None if classification fails.
        """
        try:
            # Calculate average Y-coordinates of shoulders and hips
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            hip_y = (kpts[11][1] + kpts[12][1]) / 2
            
            # Compare vertical and horizontal distance between shoulders and hips
            delta_y = abs(shoulder_y - hip_y)
            delta_x = abs((kpts[5][0] + kpts[6][0]) / 2 - (kpts[11][0] + kpts[12][0]) / 2)
            
            # If vertical distance is greater, assume upright posture -> squat
            return "squat" if delta_x < delta_y else "pushup"
        except:
            return None

    def get_kpts_indices(self, exercise):
        """
        Return the three keypoint indices required to compute the joint angle for the given exercise.

        Args:
            exercise (str): Either 'squat' or 'pushup'.

        Returns:
            List[int]: List of keypoint indices for angle estimation.
        """
        # Use right leg for squat: hip (12), knee (14), ankle (16)
        # Use right arm for push-up: shoulder (6), elbow (8), wrist (10)
        return [12, 14, 16] if exercise == "squat" else [6, 8, 10]

    def initialize_person(self, person_id):
        """
        Initialize tracking data structures for a new person ID if not already present.

        Args:
            person_id (int): ID of the tracked person.
        """
        # Ensure lists are large enough to hold data for current person ID
        while len(self.squat_count) <= person_id:
            self.squat_count.append(0)
            self.pushup_count.append(0)
            self.angle.append(0)
            self.stage.append("-")
            self.exercise.append(None)

    def update_stage_and_count(self, person_id, exercise, angle):
        """
        Update the stage ('up' or 'down') and count for a given person based on angle thresholds.

        Args:
            person_id (int): ID of the tracked person.
            exercise (str): Exercise type ('squat' or 'pushup').
            angle (float): Estimated joint angle.
        """
        # Detect transition from "up" to "down" to increment the repetition counter
        if angle < self.down_angle:
            if self.stage[person_id] == "up":
                if exercise == "squat":
                    self.squat_count[person_id] += 1
                elif exercise == "pushup":
                    self.pushup_count[person_id] += 1
            self.stage[person_id] = "down"
        elif angle > self.up_angle:
            self.stage[person_id] = "up"

    def draw_overlay(self, annotator, keypoints, kpts_indices, angle, person_id, exercise):
        """
        Annotate the image with keypoints, angle, count, and exercise stage.

        Args:
            annotator (SolutionAnnotator): Annotator instance used to draw on the frame.
            keypoints (torch.Tensor): All keypoints of the current person.
            kpts_indices (List[int]): Indices of the three keypoints used for angle calculation.
            angle (float): Computed angle value.
            person_id (int): ID of the tracked person.
            exercise (str): Type of exercise being performed.
        """
        # Highlight the 3 joints used for angle estimation
        annotator.draw_specific_kpts(keypoints, kpts_indices, radius=self.line_width * 3)
        
        # Overlay real-time info: angle, count, stage, and person ID
        annotator.plot_angle_and_count_and_stage(
            angle_text=angle,
            count_text=f"Squats:{self.squat_count[person_id]} Push-ups:{self.pushup_count[person_id]}",
            stage_text=f"{self.stage[person_id]} ({exercise}) [Person-ID {person_id}]",
            center_kpt=keypoints[kpts_indices[1]],
        )

    def process(self, im0):
        """
        Process a single image frame to detect exercises, estimate angles, count repetitions,
        and overlay visual feedback.

        Args:
            im0 (np.ndarray): The original image frame (BGR).

        Returns:
            SolutionResults: Object containing processed frame and tracking data.
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        
        # Extract pose tracking data from the image
        self.extract_tracks(im0)
        tracks = self.tracks[0]

        if tracks.boxes.id is not None:
            for i, track_id in enumerate(tracks.boxes.id.int().tolist()):
                self.initialize_person(track_id)

                # Extract keypoints for the current person
                kpts = tracks.keypoints.data[i].cpu().numpy()
                
                # Classify the exercise based on body geometry
                exercise = self.detect_exercise(kpts)
                if exercise is None:
                    continue # Skip if exercise can't be identified
                
                # Save exercise type for this person
                self.exercise[track_id] = exercise
                
                # Get the indices of keypoints used for angle estimation
                kpts_indices = self.get_kpts_indices(exercise)
                
                # Retrieve the three joints as tensors
                keypoints_cpu = [tracks.keypoints.data[i][idx].cpu() for idx in kpts_indices]

                # Estimate joint angle
                self.angle[track_id] = annotator.estimate_pose_angle(*keypoints_cpu)
                
                # Update stage and count based on the angle
                self.update_stage_and_count(track_id, exercise, self.angle[track_id])
                
                # Draw annotated overlay with info
                self.draw_overlay(
                    annotator, tracks.keypoints.data[i], kpts_indices, self.angle[track_id], track_id, exercise
                )

                # Console debug (this part can be commented): print the exercise and counts per person
                print(
                    f"[ID {track_id}] Esercizio: {exercise}, "
                    f"Squat: {self.squat_count[track_id]}, Pushup: {self.pushup_count[track_id]}"
                )

        # Final image output with annotations
        plot_im = annotator.result()
        self.display_output(plot_im)

        # Return results including frame and counts
        return SolutionResults(
            plot_im=plot_im,
            workout_count={"squat": self.squat_count, "pushup": self.pushup_count},
            workout_stage=self.stage,
            workout_angle=self.angle,
            total_tracks=len(self.track_ids),
        )
