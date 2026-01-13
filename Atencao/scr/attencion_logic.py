import time

class AttentionLogic:
    def __init__(
        self,
        ear_threshold=0.22,
        max_eye_closed_time=2.0,
        yaw_threshold=20,
        pitch_threshold=-15,
        max_head_turn_time=2.0
    ):
        self.ear_threshold = ear_threshold
        self.max_eye_closed_time = max_eye_closed_time
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.max_head_turn_time = max_head_turn_time

        self.eye_closed_start = None
        self.head_turned_start = None

        self.total_time = 0.0
        self.time_atento = 0.0
        self.time_distraido = 0.0
        self.last_update = time.time()

    def update(self, ear, pitch, yaw):
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        status = "ATENTO"

        if ear < self.ear_threshold:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
            elif now - self.eye_closed_start > self.max_eye_closed_time:
                status = "DISTRAIDO"
        else:
            self.eye_closed_start = None

       
        if pitch is not None and yaw is not None:
            if abs(yaw) > self.yaw_threshold or pitch < self.pitch_threshold:
                if self.head_turned_start is None:
                    self.head_turned_start = now
                elif now - self.head_turned_start > self.max_head_turn_time:
                    status = "DISTRAIDO"
            else:
                self.head_turned_start = None

        
        self.total_time += elapsed
        if status == "ATENTO":
            self.time_atento += elapsed
        else:
            self.time_distraido += elapsed

        percent_atento = (self.time_atento / self.total_time * 100) if self.total_time > 0 else 0
        percent_distraido = (self.time_distraido / self.total_time * 100) if self.total_time > 0 else 0

        return status, percent_atento, percent_distraido
