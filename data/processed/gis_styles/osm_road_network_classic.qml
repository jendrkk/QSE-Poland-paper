<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.34.0" simplifyDrawingHints="1" simplifyMaxScale="1" simplifyDrawingTol="1" simplifyLocal="1" simplifyAlgorithm="0" readOnly="0" labelsEnabled="0" hasScaleBasedVisibilityFlag="0">
  <renderer-v2 attr="highway" symbollevels="0" type="RuleRenderer" forceraster="0" enableorderby="0" referencescale="-1">
    <rules key="{rules-root}">
      <rule symbol="0" key="{r-other}" filter="ELSE" label="Other / unclassified"/>
      <rule symbol="1" key="{r-steps}" filter="&quot;highway&quot; = 'steps'" label="Steps" scalemindenom="25000"/>
      <rule symbol="2" key="{r-bridleway}" filter="&quot;highway&quot; = 'bridleway'" label="Bridleway" scalemindenom="25000"/>
      <rule symbol="3" key="{r-cycleway}" filter="&quot;highway&quot; = 'cycleway'" label="Cycleway" scalemindenom="25000"/>
      <rule symbol="4" key="{r-path}" filter="&quot;highway&quot; IN ('path','footway')" label="Path / Footway" scalemindenom="25000"/>
      <rule symbol="5" key="{r-track}" filter="&quot;highway&quot; IN ('track','track_grade1','track_grade2','track_grade3','track_grade4','track_grade5')" label="Track" scalemindenom="50000"/>
      <rule symbol="6" key="{r-service}" filter="&quot;highway&quot; = 'service'" label="Service" scalemindenom="50000"/>
      <rule symbol="7" key="{r-pedestrian}" filter="&quot;highway&quot; = 'pedestrian'" label="Pedestrian" scalemindenom="150000"/>
      <rule symbol="8" key="{r-livingstreet}" filter="&quot;highway&quot; = 'living_street'" label="Living street" scalemindenom="150000"/>
      <rule symbol="9" key="{r-residential}" filter="&quot;highway&quot; IN ('residential','unclassified')" label="Residential / Unclassified" scalemindenom="150000"/>
      <rule symbol="10" key="{r-tertiary}" filter="&quot;highway&quot; IN ('tertiary','tertiary_link')" label="Tertiary"/>
      <rule symbol="11" key="{r-secondary}" filter="&quot;highway&quot; IN ('secondary','secondary_link')" label="Secondary"/>
      <rule symbol="12" key="{r-primary}" filter="&quot;highway&quot; IN ('primary','primary_link')" label="Primary"/>
      <rule symbol="13" key="{r-trunk}" filter="&quot;highway&quot; IN ('trunk','trunk_link')" label="Trunk"/>
      <rule symbol="14" key="{r-motorway}" filter="&quot;highway&quot; IN ('motorway','motorway_link')" label="Motorway"/>
    </rules>
    <symbols>

      <!-- 0: Other / unclassified (catch-all, incl. NULL highway) -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="0">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="179,179,179,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.3"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 1: Steps -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="1">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="customdash" value="1;1"/>
            <Option type="QString" name="customdash_unit" value="MM"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="250,128,114,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.7"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="1"/>
          </Option>
        </layer>
      </symbol>

      <!-- 2: Bridleway -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="2">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="customdash" value="4;2"/>
            <Option type="QString" name="customdash_unit" value="MM"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="0,128,0,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.5"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="1"/>
          </Option>
        </layer>
      </symbol>

      <!-- 3: Cycleway -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="3">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="customdash" value="3;2"/>
            <Option type="QString" name="customdash_unit" value="MM"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="48,90,255,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.5"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="1"/>
          </Option>
        </layer>
      </symbol>

      <!-- 4: Path / Footway -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="4">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="customdash" value="1;3"/>
            <Option type="QString" name="customdash_unit" value="MM"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="250,128,114,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.5"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="1"/>
          </Option>
        </layer>
      </symbol>

      <!-- 5: Track -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="5">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="customdash" value="3;2"/>
            <Option type="QString" name="customdash_unit" value="MM"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="153,102,0,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.6"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="1"/>
          </Option>
        </layer>
      </symbol>

      <!-- 6: Service (thin casing + fill) -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="6">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="153,153,153,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.55"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="255,255,255,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.35"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 7: Pedestrian -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="7">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="153,153,153,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.8"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="221,221,232,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.5"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 8: Living street -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="8">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="153,153,153,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.8"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="237,237,237,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.5"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 9: Residential / Unclassified -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="9">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="153,153,153,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.8"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="255,255,255,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.5"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 10: Tertiary -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="10">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="143,143,143,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.9"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="255,255,255,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.6"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 11: Secondary -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="11">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="112,125,5,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="1.0"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="247,250,191,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.7"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 12: Primary -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="12">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="160,107,0,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="1.2"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="252,214,164,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.8"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 13: Trunk -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="13">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="200,78,47,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="1.3"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="249,178,156,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="0.9"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

      <!-- 14: Motorway -->
      <symbol force_rhr="0" type="line" alpha="1" clip_to_extent="1" name="14">
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="220,42,103,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="1.4"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
        <layer pass="0" enabled="1" locked="0" class="SimpleLine">
          <Option type="Map">
            <Option type="QString" name="capstyle" value="round"/>
            <Option type="QString" name="joinstyle" value="round"/>
            <Option type="QString" name="line_color" value="232,146,162,255"/>
            <Option type="QString" name="line_style" value="solid"/>
            <Option type="QString" name="line_width" value="1.0"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="use_custom_dash" value="0"/>
          </Option>
        </layer>
      </symbol>

    </symbols>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerOpacity>1</layerOpacity>
</qgis>
