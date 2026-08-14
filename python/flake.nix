# flake.nix
{
  description = "An (impure) flake for Python development.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

    fhs = pkgs.buildFHSEnv {
      name = "python-fhs-env";
      targetPkgs = pkgs: with pkgs; (
        [
          acl
          attr
          bzip2
          cairo
          cmake
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          cudaPackages.cuda_nvrtc
          cudaPackages.libcufft
          cudaPackages.libcutensor
          cudatoolkit
          curl
          ffmpeg_6-full
          gcc
          glib
          glibc
          graphviz  # needed by pprof --web
          kdePackages.kcachegrind
          libGL
          libssh
          libsodium
          libx11
          libxcb
          libxi
          libxml2
          openssl
          pkg-config  # for opencv2
          pprof
          ruff
          stdenv.cc.cc  # for opencv2
          systemd
          tesseract4
          ty
          util-linux
          uv
          xz
          zlib
          zstd
        ]
      );
      runScript = "bash --noprofile --norc";
      profile = ''
        export LIBRARY_PATH="/usr/lib64"
        export PYTHONPATH="${builtins.toString ./.};./."
        export MPLBACKEND="TkAgg"
        export DISPLAY=":0.0"
        export PS1="\n\[\033[0;32m\][\[\e]0;\u@\h:\w\a\]\u@nix-shell@\h:\w]\$ \[\033[0m\]"
        uv venv .venv --allow-existing --managed-python  # This should already exist if created with: uv init --managed-python --build-backend hatch macromax
        uv lock --upgrade
        uv sync --frozen --extra torch
        source .venv/bin/activate
        echo "Welcome to the $(python -V) FHS/venv shell, with interpreter $(which python)"
        code .
      '';
    };
  in
    {
      devShells.${system} = rec {
        develop = fhs.env;
        default = develop;
      };
    };
}
