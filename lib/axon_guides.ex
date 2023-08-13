Nx.Defn.default_options(compiler: EXLA)
Nx.global_default_backend(EXLA.Backend)

defmodule AxonGuides do
  @moduledoc """
  Documentation for `AxonGuides`.
  """

  def get_text! do
    case File.read("./lib/alice.txt") do
      {:ok, body} -> 
        limit = (String.length(body) / 4) |> round
        body = body |> String.slice(0..limit)

        body
        |> String.downcase()
        |> String.replace(~r/[^a-z \.\n]/, "")
        |> String.to_charlist()
      {:error, reason} -> IO.puts "Could not read file: #{reason}"
    end
  end

  @doc """
  Hello world.

  ## Examples

      iex> AxonGuides.hello()
      :world

  """
  def main do
    text = get_text!()
    characters = text |> Enum.uniq() |> Enum.sort()
    characters_count = Enum.count(characters)

    char_to_idx = characters |> Enum.with_index() |> Map.new()
    idx_to_char = characters |> Enum.with_index(&{&2, &1}) |> Map.new()

    IO.puts "Total characters: #{Enum.count(text)}"
    IO.puts "Unique characters: #{Enum.count(characters)}"

    seq_len = 10

    train_data =
      text
      |> Enum.map(&Map.fetch!(char_to_idx, &1))
      |> Enum.chunk_every(seq_len, 1, :discard)
      |> Enum.drop(-1)
      |> Nx.tensor()
      |> Nx.divide(characters_count)
      |> Nx.reshape({:auto, seq_len, 1})

    train_results =
      text
      |> Enum.drop(seq_len)
      |> Enum.map(&Map.fetch!(char_to_idx, &1))
      |> Nx.tensor()
      |> Nx.reshape({:auto, 1})
      |> Nx.equal(Nx.iota({1, characters_count}))

    model =
      Axon.input("input_chars", shape: {nil, seq_len, 1})
      |> Axon.lstm(256)
      |> then(fn {out, _} -> out end)
      |> Axon.nx(fn t -> t[[0..-1//1, -1]] end)
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(characters_count, activation: :softmax)

    batch_size = 128
    train_batches = Nx.to_batched(train_data, batch_size)
    result_batches = Nx.to_batched(train_results, batch_size)

    IO.puts("Total batches: #{Enum.count(train_batches)}")

    params =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(0.001))
      |> Axon.Loop.run(Stream.zip(train_batches, result_batches), %{}, epochs: 3, compiler: EXLA)

    generate_fn = fn model, params, init_seq ->
      init_seq = init_seq
        |> String.trim()
        |> String.downcase()
        |> String.to_charlist()
        |> Enum.map(&Map.fetch!(char_to_idx, &1))

      Enum.reduce(1..100, init_seq, fn _, seq -> 
        init_seq = seq
          |> Enum.take(-seq_len)
          |> Nx.tensor()
          |> Nx.divide(characters_count)
          |> Nx.reshape({:auto, seq_len, 1})

        char =
          Axon.predict(model, params, init_seq)
          |> Nx.argmax()
          |> Nx.to_number()

        seq ++ [char]
      end)
      |> Enum.map(&Map.fetch!(idx_to_char, &1))
    end

    init_seq = """
      not like to drop the jar for fear
    """

    generate_fn.(model, params, init_seq) |> IO.puts()
  end
end
