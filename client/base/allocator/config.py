from ...config import ClientConfig

class AllocatorConfig(ClientConfig):
    base_url: str
    timeout_s: float = 120

    path_room_alloc: str = "/gs/allocate"

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
