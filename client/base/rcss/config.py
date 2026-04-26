from typing import Any

from ...config import ClientConfig

class TrainerConfig(ClientConfig):
    prefix: str = "/trainer"

    @property
    def path_change_mode(self) -> str:
        return self.prefix + "/change_mode"

    @property
    def url_change_mode(self) -> str:
        return self.base_url + self.path_change_mode

    @property
    def path_check_ball(self) -> str:
        return self.prefix + "/check_ball"

    @property
    def url_check_ball(self) -> str:
        return self.base_url + self.path_check_ball

    @property
    def path_ear(self) -> str:
        return self.prefix + "/ear"

    @property
    def url_ear(self) -> str:
        return self.base_url + self.path_ear

    @property
    def path_eye(self) -> str:
        return self.prefix + "/eye"

    @property
    def url_eye(self) -> str:
        return self.base_url + self.path_eye

    @property
    def path_init(self) -> str:
        return self.prefix + "/init"

    @property
    def url_init(self) -> str:
        return self.base_url + self.path_init

    @property
    def path_look(self) -> str:
        return self.prefix + "/look"

    @property
    def url_look(self) -> str:
        return self.base_url + self.path_look

    @property
    def path_move(self) -> str:
        return self.prefix + "/move"

    @property
    def url_move(self) -> str:
        return self.base_url + self.path_move

    @property
    def path_recover(self) -> str:
        return self.prefix + "/recover"

    @property
    def url_recover(self) -> str:
        return self.base_url + self.path_recover

    @property
    def path_start(self) -> str:
        return self.prefix + "/start"

    @property
    def url_start(self) -> str:
        return self.base_url + self.path_start

    @property
    def path_team_names(self) -> str:
        return self.prefix + "/team_names"

    @property
    def url_team_names(self) -> str:
        return self.base_url + self.path_team_names

class ControllerConfig(ClientConfig):
    prefix: str = "/control"

    path_shutdown: str = "/control/shutdown"
    @property
    def url_shutdown(self) -> str:
        return self.base_url + self.path_shutdown

class MetricsConfig(ClientConfig):
    prefix: str = "/metrics"

    path_status: str = "/metrics/status"
    @property
    def url_status(self) -> str:
        return self.base_url + self.path_status

    path_health: str = "/metrics/health"
    @property
    def url_health(self) -> str:
        return self.base_url + self.path_health

    path_conn: str = "/metrics/conn"
    @property
    def url_conn(self) -> str:
        return self.base_url + self.path_conn


class RcssConfig(ClientConfig):
    path_room_alloc: str = "/gs/allocate"

    __trainer: TrainerConfig = None
    __control: ControllerConfig = None
    __metrics: MetricsConfig = None

    def model_post_init(self, context: Any, /) -> None:
        self.__trainer = TrainerConfig(base_url=self.base_url, prefix="/trainer")
        self.__control = ControllerConfig(base_url=self.base_url, prefix="/control")
        self.__metrics = MetricsConfig(base_url=self.base_url, prefix="/metrics")

    @property
    def trainer(self) -> TrainerConfig:
        return self.__trainer
    @property
    def control(self) -> ControllerConfig:
        return self.__control
    @property
    def metrics(self) -> MetricsConfig:
        return self.__metrics

    @property
    def url_room_alloc(self) -> str:
        """URL for room allocation endpoint."""
        return self.base_url + self.path_room_alloc

    path_room_drop: str = "/gs/drop"

    @property
    def url_room_drop(self) -> str:
        """URL for room drop endpoint."""
        return self.base_url + self.path_room_drop

    path_fleet_drop: str = "/fleet"

    @property
    def url_fleet_drop(self) -> str:
        """URL for fleet drop endpoint."""
        return self.base_url + self.path_fleet_drop

    path_fleet_create: str = "/fleet/create"

    @property
    def url_fleet_create(self) -> str:
        """URL for fleet create endpoint."""
        return self.base_url + self.path_fleet_create

    path_fleet_template: str = "/fleet/template"

    @property
    def url_fleet_template(self) -> str:
        """URL for fleet template endpoint."""
        return self.base_url + self.path_fleet_template

    path_fleet_template_version: str = "/fleet/template/version"

    @property
    def url_fleet_template_version(self) -> str:
        """URL for fleet template version endpoint."""
        return self.base_url + self.path_fleet_template_version
