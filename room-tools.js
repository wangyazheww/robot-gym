import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const $ = id => document.getElementById(id);
const floor = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const legNames = ['fl', 'fr', 'hl', 'hr'];

export function createRoomTools(view) {
  const {scene, camera, renderer, controls, robot, api} = view;
  const canvas = renderer.domElement;
  const draft = new THREE.Group();
  const overlays = new THREE.Group();
  const stageProps = new THREE.Group();
  scene.add(draft, overlays, stageProps);

  const boxMeshes = new Map();
  const gizmo = new TransformControls(camera, canvas);
  gizmo.setSize(0.65);
  const handles = gizmo.getHelper();
  scene.add(handles);
  gizmo.addEventListener('dragging-changed', event => { controls.enabled = !event.value; });
  gizmo.addEventListener('objectChange', () => {
    dirty = true;
    const visual = gizmo.object?.userData.visual;
    if (visual) {
      visual.position.copy(gizmo.object.position);
      visual.scale.copy(gizmo.object.scale);
      visual.rotation.copy(gizmo.object.rotation);
    }
    if (object?.xy) positionObject();
  });

  let world, calibration, walkable, start, savedId, place, object, objectMesh;
  let dirty = true, pick = null, playback = 0;
  let demoCache;
  const ray = new THREE.Raycaster();
  const loader = new GLTFLoader();
  const report = text => { $('calHint').textContent = text; };
  const task = text => { $('taskStatus').textContent = text; };
  const showCalibration = text => { $('stat-coord').textContent = text; };

  async function request(path, options) {
    const response = await fetch(api + path, options);
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.errors?.map(x => x.message).join('；') || result.detail || 'HTTP ' + response.status);
    }
    return result;
  }

  function post(path, data) {
    return request(path, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(data)});
  }

  function on(id, action, announce = report) {
    $(id).onclick = async () => {
      $(id).disabled = true;
      try { await action(); } catch (error) { announce(error.message); }
      finally { $(id).disabled = false; }
    };
  }

  function poseStart() {
    robot.position.set(...start.pos, view.robotHome.z);
    robot.rotation.set(0, 0, THREE.MathUtils.degToRad(start.yawDeg));
    view.standOnFloor();
  }

  function clearGroup(group) {
    for (const mesh of [...group.children]) {
      group.remove(mesh);
      disposeMesh(mesh);
    }
  }

  function disposeMesh(mesh) {
    mesh.traverse(part => {
      part.geometry?.dispose();
      part.material?.dispose();
    });
  }

  function updateOverlays({showcase = false} = {}) {
    clearGroup(overlays);
    if (walkable.length && !showcase) {
      const line = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(walkable.map(p => new THREE.Vector3(...p, .012))),
        new THREE.LineBasicMaterial({color:0x49e0c8}),
      );
      overlays.add(line);
    }
    const sameAsStart = place && Math.hypot(place[0]-start.pos[0],place[1]-start.pos[1]) < .02;
    const markers = showcase ? [[start.pos, 0xe3ae79]] : [
      [walkable.length ? start.pos : null, 0x38bdf8],
      [object?.xy, 0x34d399], [sameAsStart ? null : place, 0xfbbf24],
    ];
    for (const [xy, color] of markers) {
      if (!xy) continue;
      const marker = new THREE.Mesh(new THREE.RingGeometry(.117,.135,32),
        new THREE.MeshBasicMaterial({color, side:THREE.DoubleSide}));
      marker.position.set(...xy, .016);
      overlays.add(marker);
    }
  }

  function selectBox(id) {
    const mesh = boxMeshes.get(id);
    $('boxChoice').value = mesh ? id : '';
    $('boxKind').value = mesh?.userData.kind || 'obstacle';
    gizmo.detach();
    if (mesh && draft.visible) gizmo.attach(mesh);
  }

  function addBox({id, pose, size, kind, yawDeg = 0, navMarginM = 0}) {
    const material = new THREE.MeshBasicMaterial({
      color:kind === 'support' ? 0x34d399 : 0xffa345, transparent:true, opacity:.30, depthWrite:false,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), material);
    mesh.name = id;
    const outline = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({color:material.color, transparent:true, opacity:.86,
        depthTest:false, depthWrite:false}));
    outline.renderOrder = 10;
    mesh.add(outline);
    mesh.position.set(...pose);
    mesh.scale.set(...size);
    mesh.rotation.z = THREE.MathUtils.degToRad(yawDeg);
    mesh.userData.kind = kind;
    mesh.userData.navMarginM = navMarginM;
    draft.add(mesh);
    boxMeshes.set(id, mesh);
    $('boxChoice').append(new Option(id, id));
    selectBox(id);
    return mesh;
  }

  function setWorld(newWorld) {
    playback++;
    $('demoProgress').hidden = true;
    $('demoStatus').textContent = 'MuJoCo 实测回放 · 约 21 秒';
    pick = null;
    controls.enabled = true;
    world = newWorld;
    savedId = null;
    dirty = true;
    calibration = {floorZ:0, scaleReference:null, scaleCheck:null};
    walkable = [];
    start = {pos:newWorld.sample ? [view.robotHome.x, view.robotHome.y] : (newWorld.previewStart ?? [0,0]),
             yawDeg:newWorld.sample ? 180 : 0};
    $('robotYaw').value = start.yawDeg;
    place = null;
    object = null;
    if (objectMesh) scene.remove(objectMesh);
    objectMesh = null;
    selectBox('');
    clearGroup(draft);
    clearGroup(stageProps);
    boxMeshes.clear();
    $('boxChoice').replaceChildren(new Option('选择碰撞盒', ''));
    poseStart();
    updateOverlays();
    showCalibration('尺度未确认');
    report('先拾取地面和参考长度，再框选行走区。');
    task('');
  }

  function startPick(count, fromSplat, message, apply) {
    if (fromSplat && boxMeshes.size) throw new Error('请在放置碰撞盒前完成地面和尺度标定。');
    gizmo.detach();
    controls.enabled = false;
    pick = {count, fromSplat, points:[], apply};
    report(message + '（Esc 取消）');
    if (window.matchMedia('(max-width: 740px)').matches) canvas.scrollIntoView({block:'nearest'});
  }

  function cancelPick() {
    if (!pick) return;
    pick = null;
    controls.enabled = true;
    report('已取消拾取。');
  }

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') cancelPick();
  });

  canvas.addEventListener('pointerup', event => {
    if (!pick || event.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,
                                       1-(event.clientY-rect.top)/rect.height*2), camera);
    const hit = pick.fromSplat
      ? ray.intersectObject(view.splat(), false)[0]?.point
      : ray.ray.intersectPlane(floor, new THREE.Vector3());
    if (!hit) { report('没有拾取到有效位置，请换个机位再点。'); return; }
    pick.points.push(hit.clone());
    if (pick.points.length < pick.count) { report('第一点已选，请点击第二点。'); return; }
    const job = pick;
    pick = null;
    controls.enabled = true;
    try {
      job.apply(job.points);
      dirty = true;
      updateOverlays();
    } catch (error) { report(error.message); }
  });

  on('pickFloor', () => startPick(1, true, '点击地板上的一个可见位置', ([p]) => {
    view.splat().position.z -= p.z;
    calibration.floorZ = 0;
    report('所选地面已对齐到 z=0。');
  }));

  on('pickScale', () => {
    const meters = Number($('measureMeters').value);
    if (!(meters > 0)) throw new Error('请输入两点之间的实测长度。');
    startPick(2, true, '依次点击参考物的两个端点', ([a,b]) => {
      const span = a.distanceTo(b);
      const factor = meters/span;
      if (!Number.isFinite(factor) || factor < .02 || factor > 50) throw new Error('两点距离异常，请重新拾取。');
      const mesh = view.splat();
      mesh.position.multiplyScalar(factor);
      mesh.scale.multiplyScalar(factor);
      calibration.scaleReference = {length_m:meters, source:'manual', picked_length:span};
      poseStart();
      view.resetRoomCameras();
      showCalibration('Z-up · 米制预览');
      report('尺度已应用。请框选行走区，再添加碰撞盒。');
    });
  });

  on('pickWalk', () => startPick(2, false, '点击行走范围的两个对角', ([a,b]) => {
    const [x0,x1] = [Math.min(a.x,b.x), Math.max(a.x,b.x)];
    const [y0,y1] = [Math.min(a.y,b.y), Math.max(a.y,b.y)];
    if (x1-x0 < .6 || y1-y0 < .6) throw new Error('行走区太窄，请选更大的范围。');
    walkable = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]];
    report('行走区已标定。');
  }));

  on('pickRobot', () => startPick(1, false, '点击地面放置机器人的起点', ([p]) => {
    start.pos = [p.x,p.y];
    poseStart();
    report('机器人起点已设置。');
  }));
  $('robotYaw').onchange = () => {
    start.yawDeg = Number($('robotYaw').value);
    poseStart();
    dirty = true;
  };
  on('addBox', () => {
    const kind = $('boxKind').value;
    const target = controls.target;
    const size = kind === 'support' ? [.36,.26,.20] : [1.20,.55,.55];
    const id = (kind === 'support' ? 'support_' : 'box_') + Math.random().toString(16).slice(2,8);
    addBox({id, pose:[target.x,target.y,size[2]/2], size, kind});
    dirty = true;
    report('拖动箭头移动，选择「拉伸」可调整尺寸。');
  });
  $('boxChoice').onchange = () => selectBox($('boxChoice').value);
  $('boxKind').onchange = () => {
    if (!gizmo.object) return;
    const kind = $('boxKind').value;
    gizmo.object.userData.kind = kind;
    gizmo.object.material.color.setHex(kind === 'support' ? 0x34d399 : 0xffa345);
    gizmo.object.children[0].material.color.copy(gizmo.object.material.color);
    dirty = true;
    if (object?.xy) positionObject();
  };
  $('showBoxes').onchange = () => {
    draft.visible = $('showBoxes').checked;
    selectBox($('boxChoice').value);
  };
  function boxMode(mode) {
    gizmo.setMode(mode);
    gizmo.showX = mode !== 'rotate';
    gizmo.showY = mode !== 'rotate';
  }
  on('moveBox', () => boxMode('translate'));
  on('sizeBox', () => boxMode('scale'));
  on('rotateBox', () => boxMode('rotate'));
  on('removeBox', () => {
    const id = $('boxChoice').value;
    const mesh = boxMeshes.get(id);
    if (!mesh) return;
    selectBox('');
    draft.remove(mesh);
    if (mesh.userData.visual) {
      const visual = mesh.userData.visual;
      stageProps.remove(visual);
      visual.geometry.dispose();
      visual.material.dispose();
    }
    disposeMesh(mesh);
    boxMeshes.delete(id);
    $('boxChoice').querySelector('option[value="' + id + '"]').remove();
    dirty = true;
    if (object?.xy) positionObject();
  });

  function scenePayload() {
    const splat = view.splat();
    return {
      ...(savedId && savedId !== 's2_livingroom' ? {sceneId:savedId} : {}),
      worldId:world.worldId, splatPath:world.splat, panoPath:world.pano, sourceMode:world.sourceMode,
      calibration:{...calibration, walkablePolygon:walkable,
        splatTransform:{position:splat.position.toArray(), quaternion:splat.quaternion.toArray(), scale:splat.scale.x}},
      boxes:[...boxMeshes.values()].map(m => ({
        id:m.name, kind:m.userData.kind, pose:m.position.toArray(), size:m.scale.toArray(),
        yawDeg:THREE.MathUtils.radToDeg(m.rotation.z),
        navMarginM:m.userData.navMarginM,
      })),
      robotStart:start, place,
      object:object?.xy ? {assetId:object.assetId, xy:object.xy, material:$('materialChoice').value,
                          materialSource:object.source, supportId:supportAt(object.xy)?.name ?? null} : null,
    };
  }

  async function save() {
    if (!dirty && savedId) return savedId;
    if (!calibration.scaleReference) throw new Error('先用实测长度标定房间尺度。');
    if (!walkable.length) throw new Error('先框选行走区。');
    const saved = await post('/api/rooms/scenes', scenePayload());
    savedId = saved.sceneId;
    dirty = false;
    localStorage.setItem('robotgym-room:' + world.worldId, savedId);
    report('已保存场景 · 版本 ' + saved.draftVersion);
    return savedId;
  }
  on('saveRoom', save);

  function drawRoute(points, color) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      points.map(([x,y]) => new THREE.Vector3(x,y,.035))), new THREE.LineBasicMaterial({color}));
    overlays.add(line);
  }
  on('verifyRoom', async () => {
    const sceneId = await save();
    report('MuJoCo 运行两次碰撞对照…');
    const driveM = Number($('verifyMeters').value);
    const withBox = await post('/api/rooms/verify', {sceneId, driveM, withProxies:true});
    const without = await post('/api/rooms/verify', {sceneId, driveM, withProxies:false});
    updateOverlays();
    drawRoute(without.traj.map(p => p.slice(1,3)), 0x38bdf8);
    drawRoute(withBox.traj.map(p => p.slice(1,3)), 0xffa345);
    const hits = withBox.contacts.map(c => c.proxy).join(', ') || '无接触';
    const fault = withBox.start_penetrated ? '起点与碰撞盒重叠。' : '';
    report(fault + '有盒 ' + withBox.displacement_m.toFixed(2) + ' m（' + hits
      + '）；无盒 ' + without.displacement_m.toFixed(2) + ' m。' + withBox.run_id);
  });

  function supportAt(xy) {
    return [...boxMeshes.values()].filter(mesh => {
      if (mesh.userData.kind !== 'support') return false;
      const local = mesh.worldToLocal(new THREE.Vector3(...xy, mesh.position.z));
      return Math.abs(local.x) < .5 && Math.abs(local.y) < .5;
    })
      .sort((a,b) => (b.position.z+b.scale.z/2)-(a.position.z+a.scale.z/2))[0];
  }

  function positionObject() {
    if (!object?.xy || !objectMesh) return;
    const support = supportAt(object.xy);
    const surface = support ? support.position.z+support.scale.z/2 : 0;
    objectMesh.position.set(...object.xy, surface+object.height/2);
    objectMesh.quaternion.identity();
    objectMesh.visible = true;
  }

  async function setAsset(meta, options = {}) {
    const url = options.url || api + '/files/assets/' + encodeURIComponent(meta.id) + '/'
      + encodeURIComponent(meta.files.visual);
    const visible = (await loader.loadAsync(url)).scene;
    const scan = options.scanTransform;
    if (scan) {
      const [w,x,y,z] = scan.quaternion_wxyz;
      visible.quaternion.set(x,y,z,w).normalize();
      visible.position.fromArray(scan.position);
      visible.scale.setScalar(scan.scale);
    } else {
      visible.rotation.x = Math.PI/2;
      if (options.sampleSize) {
        const bounds = new THREE.Box3().setFromObject(visible);
        const size = bounds.getSize(new THREE.Vector3());
        visible.scale.multiplyScalar(options.sampleSize[1] / size.z);
      } else {
        visible.scale.setScalar(meta.transform?.scale ?? 1);
      }
      const center = new THREE.Box3().setFromObject(visible).getCenter(new THREE.Vector3());
      visible.position.sub(center);
    }
    if (objectMesh) scene.remove(objectMesh);
    objectMesh = new THREE.Group();
    objectMesh.name = 'room-task-object';
    objectMesh.add(visible);
    objectMesh.visible = false;
    scene.add(objectMesh);
    object = {assetId:meta.id, height:options.sampleSize?.[1] ?? meta.size_m[1],
              source:meta.material_source || 'default', xy:null};
    $('materialChoice').value = meta.material || 'unknown';
    dirty = true;
    task(meta.name + '已载入。' + (object.source === 'vlm' ? 'AI 建议材质可修改。' : '可手动选择材质。')
         + '选 A 点将模型放入房间。');
  }

  on('uploadObject', async () => {
    const file = $('objectFile').files[0];
    if (!file) throw new Error('请先选择一张物品照片。');
    const fd = new FormData();
    fd.append('image', file);
    const {id} = await request('/api/assets', {method:'POST', body:fd});
    for (let i=0; i<600; i++) {
      const job = await request('/api/assets/' + id);
      task(job.stage || job.status);
      if (job.status === 'ready') { await setAsset(job.meta); return; }
      if (job.status === 'failed') throw new Error(job.error || '物品生成失败');
      await sleep(1500);
    }
    throw new Error('等待物品超时，任务 ' + id);
  }, task);
  on('useMug', async () => {
    const assets = await request('/api/assets');
    const meta = assets.find(asset => asset.id === 'preset_mug')?.meta
      || assets.find(asset => ['mug','cup'].includes(asset.meta?.category))?.meta;
    if (!meta) throw new Error('没有杯子资产。请上传照片，或加载带杯子的 S2 搬运样例。');
    await setAsset(meta);
  }, task);
  $('materialChoice').onchange = () => {
    if (!object) return;
    object.source = 'manual';
    dirty = true;
    task('已人工标注接触面。保存后将应用对应的仿真摩擦。');
  };
  on('pickA', () => {
    if (!object) throw new Error('请先加载物品。');
    startPick(1, false, '在地面或支撑台上点选 A 点', ([p]) => {
      object.xy = [p.x,p.y];
      positionObject();
      task('物品 A 点已设置。');
    });
  }, task);
  on('pickB', () => startPick(1, false, '点击地面选择 B 点', ([p]) => {
    place = [p.x,p.y];
    task('放置 B 点已设置。');
  }), task);

  async function restore(sv) {
    const cal = sv.calibration;
    const tr = cal.splat_transform;
    if (tr) {
      view.splat().position.fromArray(tr.position);
      view.splat().quaternion.fromArray(tr.quaternion);
      view.splat().scale.setScalar(tr.scale);
      view.resetRoomCameras();
    }
    calibration = {floorZ:cal.floor_z, scaleReference:cal.scale_reference, scaleCheck:cal.scale_check};
    walkable = cal.walkable_polygon;
    start = {pos:sv.robot_start.pos, yawDeg:sv.robot_start.yaw_deg};
    $('robotYaw').value = start.yawDeg;
    for (const p of sv.proxies.filter(p => p.shape === 'box')) {
      const mesh = addBox({id:p.proxy_id, pose:p.pose, size:p.size.map(v => v*2),
        kind:p.kind, yawDeg:p.yaw_deg ?? 0, navMarginM:p.nav_margin_m ?? 0});
      if (sv.scene_id === 's2_livingroom' && p.proxy_id === 'stand_A') {
        const stand = new THREE.Mesh(new THREE.BoxGeometry(1,1,1),
          new THREE.MeshStandardMaterial({color:0x315967, roughness:.7}));
        stand.position.copy(mesh.position);
        stand.scale.copy(mesh.scale);
        stand.rotation.copy(mesh.rotation);
        stageProps.add(stand);
        mesh.userData.visual = stand;
      }
    }
    selectBox('');
    const inst = sv.instances.find(x => x.role === 'grasp_target');
    if (inst) {
      if (inst.asset_id === 'lux3d_mug_demo') {
        const [width,depth,height] = inst.extra?.size_xyz_m ?? [.061,.061,.10];
        const size = [width,height,depth];
        await setAsset({id:inst.asset_id, name:'样例杯子', size_m:size,
          material:inst.extra?.material, material_source:inst.extra?.material_source}, {
          url:'frontend/public/models/mug_visual.glb', sampleSize:size,
          scanTransform:inst.extra?.scan_transform,
        });
      } else {
        const {meta} = await request('/api/assets/' + encodeURIComponent(inst.asset_id));
        await setAsset({...meta, material:inst.extra?.material, material_source:inst.extra?.material_source});
      }
      object.xy = inst.pose.slice(0,2);
      positionObject();
    }
    place = Object.values(sv.regions).find(x => x.kind === 'place')?.pos ?? null;
    savedId = sv.scene_id;
    dirty = false;
    poseStart();
    updateOverlays();
    if (cal.scale_reference) showCalibration('Z-up · 米制预览');
    report('已加载场景 ' + savedId + ' · 版本 ' + sv.draft_version);
  }

  async function restoreLastScene() {
    const id = localStorage.getItem('robotgym-room:' + world.worldId);
    if (id) {
      try {
        const sv = await request('/api/rooms/scenes/' + encodeURIComponent(id));
        setWorld(world);
        await restore(sv);
      }
      catch (error) { report(error.message); }
    }
    return !!id;
  }
  on('restoreRoom', async () => {
    if (!await restoreLastScene()) report('该房间还没有保存过编辑。');
  });
  on('loadS2Task', async () => {
    await view.openSample();
    setWorld(world);
    await restore(await request('/api/rooms/scenes/s2_livingroom'));
    task('S2 已预标定。可以直接运行搬运。');
  }, task);

  on('oneClickDemo', async () => {
    $('demoStatus').textContent = '加载校准后的 S2 客厅和实测轨迹…';
    $('demoProgress').hidden = true;
    if (!demoCache) demoCache = await request('/demo/s2-carry.json');
    await view.openSample();
    setWorld(world);
    await restore(demoCache.scene);
    if (!$('chkRobot').checked) $('chkRobot').click();
    if (!$('chkSplat').checked) $('chkSplat').click();
    if ($('chkGrid').checked) $('chkGrid').click();
    if ($('showBoxes').checked) $('showBoxes').click();
    play(demoCache.run, {showcase:true});
  }, text => { $('demoStatus').textContent = text; });

  const nextRotation = new THREE.Quaternion();
  function applyFrame(frame, next = frame, fraction = 0) {
    const lerp = i => THREE.MathUtils.lerp(frame[i], next[i], fraction);
    const jointsBetween = (offset, count) => Array.from({length:count}, (_,i) => lerp(offset+i));
    function applyPose(node, offset) {
      node.position.set(lerp(offset),lerp(offset+1),lerp(offset+2));
      const j = offset+3;
      node.quaternion.set(frame[j+1],frame[j+2],frame[j+3],frame[j]).normalize()
        .slerp(nextRotation.set(next[j+1],next[j+2],next[j+3],next[j]).normalize(),fraction);
    }
    applyPose(robot,1);
    for (let i=0; i<legNames.length; i++) view.setDogPose(legNames[i], jointsBetween(8+i*4,4));
    view.setArmPose(jointsBetween(24,6), jointsBetween(30,2));
    if (objectMesh) applyPose(objectMesh,32);
  }

  function play(run, {showcase = false} = {}) {
    const guard = ++playback;
    updateOverlays({showcase});
    drawRoute(run.frames.map(frame => frame.slice(1,3)), 0xe3ae79);
    const startTime = performance.now();
    let lastFrameTime = startTime;
    let index = 0;
    let follow = showcase;
    const stopFollow = () => { follow = false; };
    if (follow) controls.addEventListener('start', stopFollow);
    const t = (event, fallback) => run.events.find(e => e.event === event)?.t ?? fallback;
    const end = run.frames.at(-1)[0];
    const returnHome = run.demo_mode === 'table_return';
    const segments = returnHome ? [
      {end:t('settled_at_A', end), label:'从房间中间前往茶几', speed:2},
      {end:t('grasp_result', end), label:'从茶几侧向夹起杯子', speed:1.7},
      {end, label:'保持杯口朝上，带杯返回起点', speed:2},
    ] : [
      {end:t('settled_at_A', end), label:'绕过茶几，前往取物点', speed:9},
      {end:t('grasp_result', end), label:'接近、合拢夹爪并抬起杯子', speed:2.5},
      {end:t('settled_at_B', end), label:'携带杯子，前往 B 点', speed:7},
      {end, label:'放下杯子并退爪', speed:3},
    ];
    let phaseStart = 0;
    for (const phase of segments) {
      phase.duration = Math.max(0, (phase.end-phaseStart)/phase.speed);
      phase.start = phaseStart;
      phaseStart = phase.end;
    }
    const demoProgress = $('demoProgress');
    demoProgress.hidden = !showcase;
    demoProgress.value = 0;
    const target = new THREE.Vector3();
    const eye = new THREE.Vector3();
    function frameCamera(alpha, simTime = 0) {
      const {x,y} = robot.position;
      if (returnHome) {
        const nearTable = simTime >= segments[0].end && simTime < segments[1].end;
        target.set(x+.06,y-.12,nearTable ? .46 : .39);
        eye.set(x+(nearTable ? .26 : .63),y+1.49,nearTable ? 1.22 : 1.42);
      } else {
        target.set(x+.26, y+.12, .40);
        eye.set(x+1.18, y+.83, 1.28);
      }
      controls.target.lerp(target, alpha);
      camera.position.lerp(eye, alpha);
      controls.update();
    }
    applyFrame(run.frames[0]);
    if (follow) frameCamera(1);
    const tick = time => {
      if (playback !== guard) {
        controls.removeEventListener('start', stopFollow);
        return;
      }
      let elapsed = (time-startTime)/1000;
      let simTime = elapsed*6;
      if (showcase) {
        simTime = end;
        for (const phase of segments) {
          if (elapsed < phase.duration) {
            simTime = phase.start + elapsed*phase.speed;
            if ($('demoStatus').textContent !== phase.label) $('demoStatus').textContent = phase.label;
            break;
          }
          elapsed -= phase.duration;
        }
        demoProgress.value = Math.min(1, simTime/end);
      }
      while (index+1 < run.frames.length && run.frames[index+1][0] <= simTime) index++;
      const frame = run.frames[index], next = run.frames[index+1] ?? frame;
      const span = next[0]-frame[0];
      applyFrame(frame,next,span > 0 ? THREE.MathUtils.clamp((simTime-frame[0])/span,0,1) : 0);
      if (follow) frameCamera(index+1 === run.frames.length ? 1 :
        1 - Math.exp(-(time-lastFrameTime)/210), simTime);
      lastFrameTime = time;
      if (index+1 < run.frames.length) requestAnimationFrame(tick);
      else {
        controls.removeEventListener('start', stopFollow);
        const result = run.task_success ? (run.returned_with_cup ? '已带杯回到起点' : '搬运完成')
          : '搬运未完成 · ' + (run.failure_detail || run.failure_reason);
        if (showcase) $('demoStatus').textContent = result + ' · MuJoCo 实测回放';
        task(result
          + '。' + run.disclosure + ' ' + run.run_id);
      }
    };
    requestAnimationFrame(tick);
  }

  on('runCarry', async () => {
    if (!object?.xy || !place) throw new Error('先放置物品 A 点并选定 B 点。');
    const sceneId = await save();
    const requestedWorld = world.worldId;
    task('MuJoCo 正在计算。完成后回放机器人和物品的真实轨迹。');
    const run = await post('/api/rooms/carry', {sceneId});
    if (world.worldId !== requestedWorld) return;
    task('MuJoCo 回放 · 6× · 伺服辅助、接触确认后焊接辅助');
    play(run);
  }, task);

  return {setWorld, restoreLastScene, cancelPick};
}
