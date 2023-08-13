defmodule AxonGuidesTest do
  use ExUnit.Case
  doctest AxonGuides

  test "greets the world" do
    assert AxonGuides.hello() == :world
  end
end
