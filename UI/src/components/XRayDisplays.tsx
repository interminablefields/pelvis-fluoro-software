/*
    Component to display all visualizations related to the X-ray image. Incorporates ErrorBar.tsx,
    WireSegment.tsx, Corridor.tsx, BoneOverlay.tsx based on input from the radio buttons/checkboxes.
*/
import { useEffect, useState, useRef, RefObject } from "react";
import io from "socket.io-client";
import { Image, Spinner, Flex, Box, Text } from '@chakra-ui/react';
import { useData } from '../utility/useContext';
import { ServerData, Wire, ClosestPoints } from '../utility/interfaces';
import WireSegment from "./WireSegment";
import ErrorBar from "./ErrorBar";
import BoneOverlay from "./BoneOverlay";
import Corridor from "./Corridor";

// Function to get angle formed by line segment with endpoints coord1, coord2 with x-axis.
function getYawAngle(coord1: [number, number], coord2: [number, number]) {
    const dy = coord2[1] - coord1[1];
    const dx = coord2[0] - coord1[0];
    return Math.atan2(dy, dx) * (180 / Math.PI);
}

// Function to get midpoint of line segment.
function getMidpoint(start: [number, number], end: [number, number]): [number, number] {
    return [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
}

function XRayDisplays() {
    const [imageSrc, setImageSrc] = useState("");
    const [isLoading, setIsLoading] = useState(true);
    const [serverData, setServerData] = useState<ServerData | null>(null);
    const [wireCoords, setWireCoords] = useState<[[number, number], [number, number]] | null>(null);
    const [wireAngle, setWireAngle] = useState<number | null>(null);
    const [wireMidpt, setWireMidpt] = useState<[number, number] | null>(null);
    const [corridorCoords, setCorridorCoords] = useState<[[number, number], [number, number]] | null>(null);
    const [closestDistance, setClosestDistance] = useState<number | null>(null);
    const [wcAngle, setWCAngle] = useState<number | null>(null);
    const imageRef = useRef(null);
    const { selectedWire, setSelectedWire, selectedCorridor, patientView, 
            showErrBars, showAnatomy, showWireSelect, showCorridor, showImage } = useData();
    const [closestPoints, setClosestPoints] = useState<ClosestPoints | null>(null);

    useEffect(() => {
        const socket = io("http://localhost:8000");

        socket.on("connect", () => {
            setIsLoading(true);
            console.log("connected");
        });

        socket.on("xray_data", (server_data) => {
            try {
                const data = JSON.parse(server_data);
                setServerData(data);
                setIsLoading(false);
            } catch (error) {
                console.log("Error parsing server data:", error);
            }
        });

        socket.on("connect_error", (err) => {
            setIsLoading(true);
            console.log("server error:", err.message);
        });

        socket.on("disconnect", () => {
            setIsLoading(true);
            console.log("disconnected");
        });

        return () => {
            setIsLoading(true);
            socket.disconnect();
        };
    }, []);

    useEffect(() => {
        if (serverData) {
            const newImageSrc = `data:image/jpeg;base64,${serverData.img.enc}`;
            if (newImageSrc !== imageSrc) {
                setImageSrc(`data:image/jpeg;base64,${serverData.img.enc}`);

                if (selectedCorridor && serverData.coords[patientView].corridors[selectedCorridor]?.coords) {
                    setCorridorCoords(serverData.coords[patientView].corridors[selectedCorridor].coords);

                }
                const lastMidpoint = wireMidpt;
                const lastWireAngle = wireAngle;
                const lastWireCoords = wireCoords;
                let closestWireCoords = null;
                let closestWireKey = 'wire0';
                let minDiff = Infinity;

                if(!lastMidpoint || !lastWireAngle || !lastWireCoords) {
                    if(selectedWire && serverData.coords[patientView].wires[selectedWire]?.coords) {
                        const selWireCoords = serverData.coords[patientView].wires[selectedWire].coords;
                        setWireCoords(selWireCoords);
                        setWireAngle(getYawAngle(selWireCoords[0], selWireCoords[1]));
                        setWireMidpt(getMidpoint(selWireCoords[0], selWireCoords[1]));
                    }
                }
                else {
                    Object.values(serverData.coords[patientView].wires).forEach((wire, index) => {
                        const wireMidpoint = getMidpoint(wire.coords[0], wire.coords[1]);
                        const distance = dist(lastMidpoint, wireMidpoint);
                        const angleDifference = Math.abs(lastWireAngle - getYawAngle(wire.coords[0], wire.coords[1]));
        
                        const curDiff = 3 * distance + angleDifference;
        
                        if (curDiff < minDiff) {
                            minDiff = curDiff;
                            closestWireCoords = wire.coords;
                            closestWireKey = 'wire' + index;
                        }
                    });
                    if(closestWireCoords) {
                        setWireCoords(closestWireCoords);
                        setSelectedWire(closestWireKey)
                        setWireAngle(getYawAngle(closestWireCoords[0], closestWireCoords[1]));
                        setWireMidpt(getMidpoint(closestWireCoords[0], closestWireCoords[1]));
                        console.log(closestWireKey);
                    }
                    
                }
            }
        }
    }, [imageSrc, serverData, selectedWire, wireCoords, wireAngle, setWireAngle, 
        wireMidpt, setWireMidpt, selectedCorridor, patientView, setSelectedWire]);

    const dist = (point1: [number, number], point2: [number, number]): number => {
        return Math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2);
    };

    useEffect(() => {
        const computeClosestDistance = () => {
            if (!wireCoords || !corridorCoords) {
                setClosestDistance(null);
                setClosestPoints(null);
                return;
            }
    
            let minDistance = Number.MAX_VALUE;
            let tempClosestPoints: ClosestPoints | null = null;
            console.log(wireCoords)
            for (const w of wireCoords) {
                for (const c of corridorCoords) {
                    const distance = dist(w, c);
                    if (distance < minDistance) {
                        minDistance = distance;
                        tempClosestPoints = { wirePoint: w, corridorPoint: c };
                    }
                }
            }
            setClosestDistance(minDistance);
            setClosestPoints(tempClosestPoints);
            console.log(tempClosestPoints)
        };
    
        computeClosestDistance();
    }, [wireCoords, corridorCoords]);

    useEffect(() => {
        if (selectedCorridor && serverData && serverData.coords[patientView].corridors[selectedCorridor]?.coords) {
                setCorridorCoords(serverData.coords[patientView].corridors[selectedCorridor].coords);

            }
    }, [selectedCorridor, patientView, serverData]);

    useEffect(() => {
        if (wireCoords && corridorCoords) {
            const yawWire = getYawAngle(wireCoords[0], wireCoords[1]);
            const yawCorr = getYawAngle(corridorCoords[0], corridorCoords[1]);
            setWCAngle(Math.abs(yawWire - yawCorr))
        }
    }, [wireCoords, corridorCoords]);

    
    const renderWires = (imageRef: RefObject<HTMLImageElement>) => {
        const wires = serverData?.coords[patientView]?.wires as { [key: string]: Wire };
        if (!imageSrc || !serverData || !wires || !imageRef.current) {
            return null; 
        }
        const origWidth = serverData.img.width;
        const origHeight = serverData.img.height;
        const { clientWidth: renderedWidth, clientHeight: renderedHeight } = imageRef.current;

        const scaleX = renderedWidth / origWidth;
        const scaleY = renderedHeight / origHeight;

        return (
            <svg width={renderedWidth} height={renderedHeight} style={{ position: 'absolute', top: 0, left: 0 }}>
                {Object.keys(serverData.coords[patientView].wires).map(key => {
                    const { coords } = serverData.coords[patientView].wires[key];
                    if (coords && coords.length === 2) {
                        return (
                            <WireSegment
                                key={key}
                                x1={coords[0][0] * scaleX}
                                y1={coords[0][1] * scaleY}
                                x2={coords[1][0] * scaleX}
                                y2={coords[1][1] * scaleY}
                                scaleX={scaleX}
                                scaleY={scaleY}
                                wireKey={key}
                                visible={showWireSelect}
                                getYawAngle={getYawAngle}
                                setWireCoords={setWireCoords}
                                setWireAngle={setWireAngle}
                                setWireMidpoint={setWireMidpt}
                            />
                        );
                    }
                    return null;
                })}
            </svg>
        );
    };

    const renderBones = (imageRef: RefObject<HTMLImageElement>) => {
        const curCorridor = serverData?.coords[patientView]?.corridors[selectedCorridor];
        const bones = serverData?.coords[patientView]?.bones;

        if (!imageSrc || !serverData || !bones || !imageRef.current || !curCorridor) {
            return null;
        }
    
        const origWidth = serverData.img.width;
        const origHeight = serverData.img.height;
        const { clientWidth: renderedWidth, clientHeight: renderedHeight } = imageRef.current;
    
        const scaleX = renderedWidth / origWidth;
        const scaleY = renderedHeight / origHeight;

        const boneColors = {
            "femur_left": "purple",
            "femur_right": "red",
            "hip_left": "blue",
            "hip_right": "green",
            "sacrum": "turquoise"
        };

        const displayBones = curCorridor.relevant_bones;

        return (
            <svg width={renderedWidth} height={renderedHeight} style={{ position: 'absolute', top: 0, left: 0 }}>
                {Object.keys(serverData.coords[patientView].bones).filter(key => displayBones.includes(key)).map((key) => {
                    const color = (boneColors as { [key: string]: string })[key] || 'gray';
                    const coords = bones[key].coords;
                    if (coords) {
                        const scaledCoords = coords.map(pair => [pair[0] * scaleX, pair[1] * scaleY] as [number, number]);
                        return (
                            <BoneOverlay
                                points= {scaledCoords}
                                color={color}
                                visible={showAnatomy}
                            />
                        );
                    }
                    return null;
                })}
            </svg>
        );
    };

    const renderCorridor = (imageRef: RefObject<HTMLImageElement>) => {
        if (!showCorridor || !corridorCoords || !imageSrc || !imageRef.current || !serverData) {
            return null;
        }
        const origWidth = serverData.img.width;
        const origHeight = serverData.img.height;
        const { clientWidth: renderedWidth, clientHeight: renderedHeight } = imageRef.current;

        const scaleX = renderedWidth / origWidth;
        const scaleY = renderedHeight / origHeight;

        const coords = corridorCoords;

        return (
            <svg width={renderedWidth} height={renderedHeight} style={{ position: 'absolute', top: 0, left: 0 }}>
                <Corridor
                    x1 = {coords[0][0] * scaleX}
                    y1 = {coords[0][1] * scaleY}
                    x2 = {coords[1][0] * scaleX}
                    y2 = {coords[1][1] * scaleY}
                />

            </svg>
        );
    };
    
    const renderErrorBars = (imageRef: RefObject<HTMLImageElement>) => {
        if (!closestPoints || !showErrBars) {
            return null;
        }
        const { wirePoint, corridorPoint } = closestPoints;

        if (!imageSrc || !corridorPoint || !wirePoint || !imageRef.current || !serverData) {
            return null; 
        }
        const origWidth = serverData.img.width;
        const origHeight = serverData.img.height;
        const { clientWidth: renderedWidth, clientHeight: renderedHeight } = imageRef.current;

        const scaleX = renderedWidth / origWidth;
        const scaleY = renderedHeight / origHeight;
        const wireX = wirePoint[0] * scaleX;
        const wireY = wirePoint[1] * scaleY;
        const corrX = corridorPoint[0] * scaleX;
        const corrY = corridorPoint[1] * scaleY;

        return (
            <ErrorBar
                wireX={wireX}
                wireY={wireY}
                corrX={corrX}
                corrY={corrY}
                clientWidth={renderedWidth}
                clientHeight={renderedHeight}
                visible={showErrBars}
            />
        );
    };

    return (
        <Flex height="100vh" align="center" justify="center">
            {isLoading ? (
                <Spinner size="xl" color="teal.500" thickness="4px" speed="0.65s" emptyColor="gray.200" />
            ) : (
                <Box height="100%" width="auto" overflow="hidden" maxWidth="100%" maxHeight="100%" position="relative">
                    {imageSrc && <Image ref={imageRef} src={imageSrc} alt="X-Ray Image" objectFit="contain" style={{
                    opacity: showImage ? 1 : 0.1}} />}
                    {renderBones(imageRef)}
                    {renderCorridor(imageRef)}
                    {renderErrorBars(imageRef)}
                    {renderWires(imageRef)}

                    {closestDistance && (
                        <Text position="absolute" top="1rem" right="1rem" fontSize="xl" style={{ textShadow: '0 0 5px #000000' }}>
                            Wire-to-Corridor Distance: {closestDistance.toFixed(2)} units
                        </Text>
                    )}
                    {wcAngle && (
                        <Text position="absolute" top="3rem" right="1rem" fontSize="xl" style={{ textShadow: '0 0 5px #000000' }}>
                            Wire-to-Corridor Angle Offset: {wcAngle.toFixed(2)} degrees
                        </Text>
                    )}
                </Box>
            )}
        </Flex>
    );
}

export default XRayDisplays;
