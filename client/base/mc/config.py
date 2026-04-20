from ...config import ClientConfig

class MatchComposerConfig(ClientConfig):
    path_match_start: str = "/start"

    @property
    def url_match_start(self) -> str:
        return self.base_url + self.path_match_start

    path_match_stop: str = "/stop"

    @property
    def url_match_stop(self) -> str:
        return self.base_url + self.path_match_stop

    path_match_restart: str = "/restart"

    @property
    def url_match_restart(self) -> str:
        return self.base_url + self.path_match_restart

    path_match_status: str = "/status"

    @property
    def url_match_status(self) -> str:
        return self.base_url + self.path_match_status

    path_team_status: str = "/team/status"

    @property
    def url_team_status(self) -> str:
        return self.base_url + self.path_team_status

    path_fleet_template_version: str = "/fleet/template/version"

    @property
    def url_fleet_template_version(self) -> str:
        """URL for fleet template version endpoint."""
        return self.base_url + self.path_fleet_template_version
