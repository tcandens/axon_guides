defmodule AxonGuides.MixProject do
  use Mix.Project

  def project do
    [
      app: :axon_guides,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {AxonGuides.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.3.0"},
      {:nx, "~> 0.4.0", override: true},
      {:exla, "~> 0.4.0"},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
    ]
  end
end
