# Open items

This file records the information that [`README.md`](../README.md) does not yet provide. The
README states each gap inline as **not provided** and carries no unresolved markers; this file is
the single place where the gaps are tracked.

<table align="center">
  <thead>
    <tr>
      <th>README section</th>
      <th>Missing item</th>
      <th>Required to close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="../README.md#53-step-3-run-inference-in-the-web-interface">5.3</a></td>
      <td>Reference Minimal Working Example results</td>
      <td>Observed anomaly scores, wall-clock runtime, and a capture of the completed run.</td>
    </tr>
    <tr>
      <td><a href="../README.md#82-staging-a-dataset-manually">8.2</a></td>
      <td>Acquisition details for the manually staged datasets</td>
      <td>Acquisition procedure, archive size, license, download date, and staged directory structure for RAW-FABRID, TILDA-400, the Fabric Defects Dataset, and Tianchi.</td>
    </tr>
    <tr>
      <td><a href="../README.md#85-public-dataset-sources">8.5</a></td>
      <td>Source URLs for three datasets</td>
      <td>Canonical source URL for TILDA-400, the Fabric Defects Dataset, and the Tianchi Guangdong challenge.</td>
    </tr>
    <tr>
      <td><a href="../README.md#92-downloading-the-distributed-weights">9.2</a></td>
      <td>Weight revision and checksums</td>
      <td>Pinned Hugging Face Hub revision, and the SHA-256 checksum of each of the 19 distributed files.</td>
    </tr>
    <tr>
      <td><a href="../README.md#93-training-a-general-domain-weight">9.3</a></td>
      <td>Reference general-domain training run</td>
      <td>Wall-clock time and image AUROC for <code>PatchCore</code> trained on the MVTec AD <code>bottle</code> category.</td>
    </tr>
    <tr>
      <td><a href="../README.md#105-recorded-demonstrations">10.5</a></td>
      <td>Recording revision</td>
      <td>Commit revision shown in <code>detection.mp4</code> and <code>benchmark.mp4</code>.</td>
    </tr>
    <tr>
      <td><a href="../README.md#115-benchmarking">11.5</a></td>
      <td>Example benchmark leaderboard</td>
      <td>Expected leaderboard produced by <code>configs/archive/benchmark_example.yaml</code>.</td>
    </tr>
    <tr>
      <td><a href="../README.md#19-additional-documentation">19</a></td>
      <td>Maintainer contact</td>
      <td>A public contact address for the maintainers.</td>
    </tr>
  </tbody>
</table>

Closing an item means replacing the corresponding **not provided** statement in `README.md` with
the recorded value, then deleting its row here.
