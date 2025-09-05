from locust import HttpUser, task, between
import os

class VideoUser(HttpUser):
    wait_time = between(1, 5)  # wait between requests

    @task
    def upload_video(self):
        # relative path (file must be in same folder as locustfile.py)
        test_file = os.path.join(os.path.dirname(__file__), "tests", "tes_vid.mp4")

        with open(test_file, "rb") as f:
            self.client.post(
                "/upload",
                files={"file": f},
                data={"margin": 2}
            )
