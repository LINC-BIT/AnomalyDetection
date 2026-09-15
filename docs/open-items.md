# Open items

This file records the information that [`README.md`](../README.md) does not yet provide. Each row
names the section where the missing value belongs; none of these gaps carries an unresolved marker
in the README itself, so this file is the single place where they are tracked.

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
      <td><a href="../README.md#52-run-anomaly-detection-example">5.2</a></td>
      <td>Reference example results</td>
      <td>Observed anomaly scores, wall-clock runtime, and a capture of the completed run.</td>
    </tr>
    <tr>
      <td><a href="../README.md#225-download-datasets">2.2.5</a></td>
      <td>Acquisition details for the manually staged datasets</td>
      <td>Acquisition procedure, archive size, license, download date, and staged directory structure for RAW-FABRID, TILDA-400, the Fabric Defects Dataset, and Tianchi.</td>
    </tr>
    <tr>
      <td><a href="../README.md#225-download-datasets">2.2.5</a></td>
      <td>Source URLs for three datasets</td>
      <td>Canonical source URL for TILDA-400, the Fabric Defects Dataset, and the Tianchi Guangdong challenge.</td>
    </tr>
    <tr>
      <td><a href="../README.md#226-download-checkpoints">2.2.6</a></td>
      <td>Weight revision and checksums</td>
      <td>Pinned Hugging Face Hub revision, and the SHA-256 checksum of each distributed file.</td>
    </tr>
    <tr>
      <td><a href="../README.md#422-training">4.2.2</a></td>
      <td>Reference general-domain training run</td>
      <td>Wall-clock time and image AUROC for <code>PatchCore</code> trained on the MVTec AD <code>bottle</code> category.</td>
    </tr>
    <tr>
      <td><a href="../README.md#41-web-front-end">4.1</a></td>
      <td>Recording revision</td>
      <td>Commit revision shown in <code>detection.mp4</code> and <code>benchmark.mp4</code>.</td>
    </tr>
    <tr>
      <td><a href="../README.md#425-benchmarking">4.2.5</a></td>
      <td>Example benchmark leaderboard</td>
      <td>Expected leaderboard produced by <code>configs/archive/benchmark_example.yaml</code>.</td>
    </tr>
    <tr>
      <td><a href="../README.md#6-extensibility">6</a></td>
      <td>Maintainer contact</td>
      <td>A public contact address for the maintainers.</td>
    </tr>
  </tbody>
</table>

Closing an item means writing the recorded value into the named section of `README.md`, then
deleting its row here.
