{
  description = "Flake for Python development.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      envWithScript = script:
        (pkgs.buildFHSEnv {
          name = "python-env";
          targetPkgs = pkgs: (with pkgs; [
            python3
            python3Packages.pip
            python3Packages.virtualenv
            pythonManylinuxPackages.manylinux2014Package
            cmake
            ninja
            gcc
            pre-commit
          ]);
          runScript = "${pkgs.writeShellScriptBin "runScript" (''
                     set -e
                     test -d .venv || ${pkgs.python3.interpreter} -m venv .venv
              source .venv/bin/activate
              set +e
            ''
            + script)}/bin/runScript";
        })
        .env;
    in {
      devShell = envWithScript "bash";
    });
}
