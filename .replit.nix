{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.flask
    pkgs.python310Packages.requests
    pkgs.python310Packages.gunicorn
    pkgs.python310Packages.psycopg2
  ];
  env = {
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    PYTHONHOME = "${pkgs.python310}";
  };
}
